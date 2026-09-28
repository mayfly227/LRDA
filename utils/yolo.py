from typing import List

import torch


class YoloRewardModel(torch.nn.Module):
    def __init__(self, weight_path: str):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO reward training. Install it in the training environment first."
            ) from exc

        self.yolo = YOLO(weight_path)
        print(f"Loaded YOLO model from {weight_path}")
        self.model = self.yolo.eval()

        for parameter in self.model.model.parameters():
            parameter.requires_grad_(False)

    def targetLoss(self, images: torch.Tensor):
        cls_idx = 1
        results_train = self.model(
            images, verbose=False, conf=0.000001, iou=0.5, embed=[-1]
        )
        loss = -results_train[0][:, 4 + cls_idx, :].sum()
        return True, loss

    def confidence_loss(
        self,
        images: torch.Tensor,
        mode: str = "non_targeted",
        target_class_id: int = 2,
    ) -> tuple[bool, torch.Tensor]:
        """Compute differentiable confidence-based loss from YOLO raw predictions.

        non_targeted: minimize confidence for the selected class.
        targeted: maximize confidence for the selected class by minimizing
        the negative confidence.
        """
        results_train = self.model(images, verbose=False, conf=0.01, iou=0.5, embed=[-1])
        class_scores = results_train[0][:, 4 + target_class_id, :]
        if mode == "non_targeted":
            kept_scores = class_scores[class_scores > 0.05]
            if len(kept_scores) == 0:
                return False, torch.FloatTensor([0.0]).to(device=images.device)
            loss = kept_scores.mean()
        elif mode == "non_targeted_soft":
            # Softmax-weighted mean of the car scores (no hard threshold): a smooth
            # surrogate for "the strongest car detection", differentiable everywhere so
            # the training signal never extinguishes / skips steps as scores drop past
            # the old 0.1 cut. Bounded in [0, 1] (no logsumexp +log(A) offset) so the
            # logged value is readable; gradient concentrates on the high anchors.
            # tau -> 0 approaches the true max; larger tau spreads over more anchors.
            tau = float(getattr(self, "soft_tau", 0.1))
            weights = torch.softmax(class_scores / tau, dim=1)  # (B, A)
            loss = (weights * class_scores).sum(dim=1).mean()
        elif mode == "non_targeted_all":
            all_class_scores = results_train[0][:, 4:, :]
            kept_scores = all_class_scores[all_class_scores > 0.05]
            if len(kept_scores) == 0:
                return False, torch.FloatTensor([0.0]).to(device=images.device)
            loss = kept_scores.mean()
        elif mode == "non_targeted_nms":
            # Minimize the single strongest detection -- the box NMS would actually report.
            # NMS never removes the top-scoring box (it only suppresses overlapping lower-conf
            # duplicates), so the per-image max class score IS the confidence of the
            # "largest box after NMS". Driving it down pushes that one detection below the
            # threshold so the car becomes undetected. (Equivalent to -targeted.)
            loss = class_scores.amax(dim=1).mean()
        elif mode == "targeted":
            loss = -class_scores.amax(dim=1).mean()
        else:
            raise ValueError(
                f"Unknown YOLO reward mode: {mode}. Expected 'non_targeted', "
                "'non_targeted_soft', 'non_targeted_all', 'non_targeted_nms', or 'targeted'."
            )

        return True, loss

    def configure_fabricate(
        self,
        orig_class_id: int = 2,
        grid_size: int = 5,
        num_targets: int = 3,
        conf_target: float = 0.6,
        suppress_weight: float = 1.0,
        fab_weight: float = 1.0,
        tau: float = 0.1,
        hinge: str = "squared",
    ) -> None:
        """Store hyper-parameters for `fabricate_loss` so the positional
        `forward(images, mode, target_class_id)` call site stays unchanged."""
        self.fab_cfg = {
            "orig_class_id": int(orig_class_id),
            "grid_size": int(grid_size),
            "num_targets": int(num_targets),
            "conf_target": float(conf_target),
            "suppress_weight": float(suppress_weight),
            "fab_weight": float(fab_weight),
            "tau": float(tau),
            "hinge": str(hinge),
        }

    def fabricate_loss(
        self,
        images: torch.Tensor,
        target_class_id: int,
        orig_class_id: int = 2,
        grid_size: int = 5,
        num_targets: int = 3,
        conf_target: float = 0.6,
        suppress_weight: float = 1.0,
        fab_weight: float = 1.0,
        tau: float = 0.1,
        hinge: str = "squared",
    ) -> tuple[bool, torch.Tensor]:
        """Targeted "phantom object" loss (lower == stronger attack).

        Semantics align with `targeted`: `target_class_id` is the class we want
        the detector to report. Two parts:
          1. The original class (`orig_class_id`, e.g. the car) must NOT be
             detected -> drive its strongest anchor confidence down.
          2. `target_class_id` objects should appear *where the car is* (the only
             optimizable region): tile the frame into a grid, pick the
             `num_targets` cells the car currently occupies, and push the target
             confidence up in each. This concentrates gradient on the car body
             and disperses the phantoms across it.

        Returns a single scalar so it plugs into the existing per-view reward
        aggregation in `compute_multi_view_reward_loss`.
        """
        _, _, height, width = images.shape
        results_train = self.model(
            images, verbose=False, conf=0.01, iou=0.5, embed=[-1]
        )
        pred = results_train[0]  # (B, 4 + nc, A)
        boxes = pred[:, :4, :]  # (B, 4, A) xywh in input pixels
        scores = pred[:, 4:, :]  # (B, nc, A) per-class probability
        batch, num_classes, _ = scores.shape
        for name, class_id in (
            ("target_class_id", target_class_id),
            ("orig_class_id", orig_class_id),
        ):
            if not 0 <= class_id < num_classes:
                raise ValueError(
                    f"{name}={class_id} out of range for nc={num_classes}."
                )

        # ---- box centers -> normalized grid cell (assignment detached) ----
        cx = (boxes[:, 0, :] / width).clamp(0.0, 1.0 - 1e-6)
        cy = (boxes[:, 1, :] / height).clamp(0.0, 1.0 - 1e-6)
        gx = (cx.detach() * grid_size).long().clamp(0, grid_size - 1)
        gy = (cy.detach() * grid_size).long().clamp(0, grid_size - 1)
        cell = gy * grid_size + gx  # (B, A) in [0, grid_size**2)

        orig_scores = scores[:, orig_class_id, :]  # (B, A)
        target_scores = scores[:, target_class_id, :]  # (B, A)

        # ---- (1) suppress the original class everywhere (smooth max) ----
        suppress = (tau * torch.logsumexp(orig_scores / tau, dim=1)).mean()

        # ---- (2) fabricate target objects on the car's grid cells ----
        # For each occupied cell collect a differentiable target confidence
        # (soft max over anchors) and a detached selection score. Selection
        # follows max(car, target): it locks onto the car at first and stays on
        # the phantom once it is established, both of which sit on car pixels.
        cell_target = []
        cell_select = []
        for b in range(batch):
            for c in range(grid_size * grid_size):
                mask = cell[b] == c
                if not bool(mask.any()):
                    continue
                cell_conf = tau * torch.logsumexp(target_scores[b][mask] / tau, dim=0)
                cell_target.append(cell_conf)
                cell_select.append(
                    torch.maximum(
                        orig_scores[b][mask].max(),
                        target_scores[b][mask].max(),
                    ).detach()
                )

        if not cell_target:
            zero = scores.new_zeros(())
            return True, suppress_weight * suppress + zero

        cell_target = torch.stack(cell_target)  # (num_occupied_cells,)
        cell_select = torch.stack(cell_select)
        k = min(int(num_targets), cell_target.numel())
        top_idx = torch.topk(cell_select, k=k, largest=True).indices
        selected = cell_target[top_idx]

        excess = (conf_target - selected).clamp(min=0.0)
        fab = excess.pow(2).mean() if hinge == "squared" else excess.mean()

        loss = suppress_weight * suppress + fab_weight * fab
        return True, loss

    @torch.no_grad()
    def annotate(
        self,
        images: torch.Tensor,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> list:
        """Run normal (NMS) prediction and return RGB uint8 images with the
        detection boxes/labels drawn, one per input image."""
        import numpy as np

        results = self.model.predict(images, verbose=False, conf=conf, iou=iou,embed=None)
        annotated = []
        for res in results:
            bgr = res.plot()  # HWC, BGR, uint8
            annotated.append(np.ascontiguousarray(bgr[:, :, ::-1]))  # -> RGB
        return annotated

    def maxProExtract(
        self,
        images: torch.Tensor,
        class_id: List[int],
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> tuple[bool, torch.Tensor]:
        device = images.device
        if len(class_id) == 0:
            class_id = list(range(80))
        results_train = self.model(images, verbose=False, conf=conf, iou=iou)
        loss1 = torch.FloatTensor([0.0]).to(device=device)
        for res1 in results_train:
            for box in res1.boxes:
                conf = box.conf
                clas = box.cls
                if len(conf) == 0:
                    continue
                for con, c in zip(conf, clas):
                    if con > 0 and c in class_id:
                        # if con > 0:
                        loss1 = loss1 + con
        if loss1.item() < 0.00001:
            return False, torch.FloatTensor([0.0]).to(device=device)
        return True, loss1

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "non_targeted",
        target_class_id: int = 0,
    ) -> tuple[bool, torch.Tensor]:
        # return self.maxProExtract(
        #     images,
        #     [target_class_id])
        if mode == "fabricate":
            cfg = getattr(self, "fab_cfg", {})
            return self.fabricate_loss(
                images,
                target_class_id=target_class_id,
                **cfg,
            )
        return self.confidence_loss(
            images,
            mode=mode,
            target_class_id=target_class_id,
        )
