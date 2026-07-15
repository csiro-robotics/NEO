# Paper Reproducibility

## Reference Environment

The local `gs` environment used during release preparation contains:

- Python 3.10.16, PyTorch 2.6.0, torchvision 0.21.0;
- Nerfstudio 1.1.5 from commit `d8b04cfaa30422de1189ee987ba7d3950b55a650`;
- LERF fork commit `112ad9ede65c595e6c0f1f9a675f15891665c4c4`;
- NeRFiller fork commit `31666136e7d4e9770652185a890484dabfb2979d`;
- Diffusers 0.32.2 and Transformers 4.49.0.

The exact GPU model, driver, CUDA toolkit build, random seeds, and runtime logs
must be added to the final artifact manifest.

## Experiment Sets

Removal scenes: `mustard_bottle`, `orange`, `red_robot`, `spam`, and
`tomato_soup_can`.

Relocation scenes: `banana`, `orange`, `rubiks_cube`, `spam`, `sugar`, and
`tomato_soup_can`.

Assembly sequence: `lego_bricks_1`, `red_cup`, `lego_bricks_2`, and
`golf_ball`. The tracked assembly manifest executes all four steps in order.

## Evaluation Defaults

The current release evaluator uses a frame separation of 20, pixel stride 4,
depth scale 0.001, maximum depth 2.0 m, and depth-consistency threshold 0.05 m.
Its reported `Reproj Mean` is the mean RGB L1 error after projection. A separate
two-way round-trip pixel reprojection error is also stored in the JSON output
but is not placed in the summary table.

## Items Requiring Author Resolution

1. The density-zero regularizer described in paper Eqs. (4)-(5) is currently
   commented out in `SparseGuidedNeRFModel.get_loss_dict`.
2. Table II reports NEO depth RMSE `0.141/0.182`, while Table III reports
   `0.056/0.033` for the full pick-and-place model.
3. The paper must state explicitly that `E_rep` is the photometric RGB L1
   statistic if that is the intended table value.
4. DFF and Seal3D code or externally generated result manifests are required
   to reproduce every baseline row.
5. Expected aggregate JSON/CSV files and random seeds must be checked in after
   the final reference runs are selected.

Do not claim exact table reproduction until these five items are resolved.
