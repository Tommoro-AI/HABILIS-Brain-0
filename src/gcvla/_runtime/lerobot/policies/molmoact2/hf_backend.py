# Modifications by Tommoro: public runtime extraction and adaptation.
# This file differs from its original source; see manifest.json for source hashes.

class MolmoAct2HFBackend:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("The policy requires a resolved local model directory")
