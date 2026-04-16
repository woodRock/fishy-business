# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTMixin:
    """
    Mixin to add Test-Time Training (entropy minimization) capability to any PyTorch model.
    """

    def forward_ttt(
        self, x: torch.Tensor, lr: float = 1e-3, steps: int = 1
    ) -> torch.Tensor:
        """
        Test-Time Training (TTT) via Entropy Minimization.
        Adapts the model to the specific sample x at inference time.
        """
        was_training = self.training
        self.train()  # Enable dropout/grads if desired, but we usually want eval mode for TTT consistency

        # Optimize Norms and Gains (TTT-Lite style)
        # These parameters are most sensitive to distribution shifts.
        ttt_params = []
        for n, p in self.named_parameters():
            if any(
                k in n.lower() for k in ["norm", "gain", "scale", "mix", "ln_", "bn_"]
            ):
                p.requires_grad = True
                ttt_params.append(p)
            else:
                p.requires_grad = False

        # Fallback to all parameters if no norm layers found
        if not ttt_params:
            for p in self.parameters():
                p.requires_grad = True
            ttt_params = list(self.parameters())

        optimizer = torch.optim.SGD(ttt_params, lr=lr)

        # TTT MUST run with gradients enabled
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad()
                logits = self(x)
                if isinstance(logits, tuple):
                    logits = logits[0]

                probs = F.softmax(logits, dim=-1)
                # Minimize entropy -> make the model more confident
                entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
                entropy.backward()
                optimizer.step()

        # Reset requires_grad for future standard training
        for p in self.parameters():
            p.requires_grad = True

        self.eval()
        with torch.no_grad():
            final_logits = self(x)
            if isinstance(final_logits, tuple):
                final_logits = final_logits[0]

        self.train(was_training)
        return final_logits
