# External Dependencies

Inference source derived from the Molmo/MolmoAct2 codebase and LeRobot is vendored
under `src/gcvla/_runtime`. Original licenses are included as `MOLMOACT2-LICENSE`,
`OLMO-LICENSE` and `LEROBOT-LICENSE`. The manifest records original and redistributed source hashes.
The included source is adapted for GC-VLA inference and evaluation.

The following dependencies are not vendored as source repositories in this source
tree. This does not mean their weights are absent from a derived checkpoint:
GC-VLM is initialized from Molmo2-ER, as disclosed above.

| Component | Upstream | Purpose |
| --- | --- | --- |
| Molmo2-ER | https://huggingface.co/allenai/Molmo2-ER | GC-VLM initialization |
| DINOv2 | https://github.com/facebookresearch/dinov2 | GC feature generation |
| Depth Anything 3 | https://github.com/ByteDance-Seed/Depth-Anything-3 | GC geometry generation |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | Evaluation benchmark |

Pin each checkout and model revision in a local release lock before running. Review
code and weight licenses separately; a repository's code license does not necessarily
cover its model weights.

The [Molmo2-ER model card](https://huggingface.co/allenai/Molmo2-ER) separately
identifies its weights as Apache-2.0. This is the VLM initialization dependency;
code reuse from MolmoAct2 does not imply use of MolmoAct2 action-policy weights.
The checkpoint distribution must accompany its weights with upstream attribution
and license text, even when hosted separately from this source repository.
