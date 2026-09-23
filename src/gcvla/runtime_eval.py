"""Entrypoint for the bundled runtime ABI."""

import re
from pathlib import Path

BASE_SUITES = frozenset({"libero_spatial", "libero_object", "libero_goal", "libero_10"})


def is_pro_suite(name):
    return any(
        name == f"{base}_{axis}"
        for base in BASE_SUITES
        for axis in ("lan", "object", "swap", "task")
    )


def main():
    from runtime_compat import register_public_model_name

    register_public_model_name()
    import lerobot.envs.libero as env
    from lerobot.scripts.lerobot_eval import main as evaluate

    original = env._get_suite

    def get_suite(name):
        suite = original(name)
        if is_pro_suite(name):
            for index, task in enumerate(suite.tasks):
                path = (
                    Path(env.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                )
                match = re.search(r"\(:language\s+([^)]*)\)", path.read_text())
                if not match:
                    raise ValueError(f"Missing BDDL instruction: {path}")
                suite.tasks[index] = task._replace(language=" ".join(match[1].split()))
        return suite

    env._get_suite = get_suite
    for suite, limit in list(env.TASK_SUITE_MAX_STEPS.items()):
        for axis in ("lan", "object", "swap", "task"):
            env.TASK_SUITE_MAX_STEPS[f"{suite}_{axis}"] = limit
    evaluate()


if __name__ == "__main__":
    main()
