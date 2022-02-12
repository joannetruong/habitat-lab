#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.registry import registry
from habitat.core.simulator import Simulator

def _try_register_igibson_socialnav():
    try:
        import habitat_sim  # noqa: F401

        has_habitat_sim = True
    except ImportError as e:
        has_habitat_sim = False
        habitat_sim_import_error = e

    if has_habitat_sim:
        from habitat.sims.igibson_challenge.social_nav import (
            iGibsonSocialNav
        )  # noqa: F401
        from habitat.sims.igibson_challenge.interactive_nav import (
            iGibsonInteractiveNav
        )  # noqa: F401
    else:
        @registry.register_simulator(name="iGibsonSocialNav")
        class iGibsonSocialNavImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise habitat_sim_import_error
        @registry.register_simulator(name="iGibsonInteractiveNav")
        class iGibsonSocialNavImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise habitat_sim_import_error
