CJKVI IDS optional data
=======================

HanziStyleForge does not bundle ids.txt.

The program downloads a pinned copy directly from:
https://github.com/cjkvi/cjkvi-ids

Automatic installation occurs when the component atlas or component-aware
refinement first needs the file. You can install it explicitly with:

.venv\Scripts\python.exe hanzistyleforge.py --config config_fusion_months_12gb.json ids-install

Installed files:

data\cjkvi-ids\ids.txt
data\cjkvi-ids\source.json

The pinned upstream revision is:
86b4d16159f0079437870408f0ca186e529015db

Expected ids.txt SHA-256:
bfc70a8c09f9f5616ebf0543bd6681e67314e9f7ae2307e5ae8c6f15bdc5c6a6

Licensing
---------

The cjkvi/cjkvi-ids README states that ids.txt is derived from the CHISE
project and follows the applicable CHISE terms. The downloaded file is not
covered by the HanziStyleForge software license. Review the upstream terms
before copying or redistributing ids.txt.

The file is optional. If it is unavailable, HanziStyleForge continues without
semantic component residual hints and uses its other generation and refinement
paths.
