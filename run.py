from argparse import ArgumentParser

import time
from utils import Display3d, parse_results, get_space


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--nsteps", type=int, default=100000)
    parser.add_argument("--nbodies", type=int, default=8)
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--v", type=float, default=300)
    parser.add_argument("--G", type=float, default=2e3)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=1)
    parser.add_argument("--softening", type=float, default=0.01)
    parser.add_argument("--object_scale", type=float, default=1.5)
    parser.add_argument("--camera_distance", type=float, default=None)
    parser.add_argument("--write_interval", type=int, default=10)
    parser.add_argument("--trajectories", default=None)
    parser.add_argument(
        "--backend",
        choices=["nbody_cuda", "nbody_c", "nbody_pytorch", "nbody_triton"],
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Will not write trajectories, used to compare run times",
    )
    args = parser.parse_args()
    if args.camera_distance is None:
        args.camera_distance = args.r + args.v
    print(args)
    return args


def main():
    args = parse_args()
    backend = __import__(args.backend)
    if args.trajectories is None:
        space = get_space(args)
        t0 = time.time()
        backend.run(
            space,
            args.nsteps,
            args.G,
            args.dt,
            args.damping,
            args.softening,
            args.write_interval if not args.benchmark else args.nsteps,
        )
        print("time:", time.time() - t0)
        trajectories_path = "trajectories/results.data"
    else:
        trajectories_path = args.trajectories
    if args.render:
        trajectories = parse_results(trajectories_path)
        app = Display3d(
            trajectories,
            camera_position=[0, args.camera_distance, 0],
            object_scale=args.object_scale,
        )
        app.run()


if __name__ == "__main__":
    main()
