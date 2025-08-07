try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch
from scipy.spatial.transform import Rotation

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=True,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)


args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere, cone

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction
import omni.isaac.core.utils.prims as prim_utils

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
import time
import pickle

############################################################


########### OV #################;;;;;


def main():
    EXECUTION_FREQ = 50
    TRAJOPT_STEPS = 64
    FLOOR_TO_RAIL_POSITION = 0.328 + 0.325
    TRAJECTORY_OUTPUT_PATH = "/home/antonio/Downloads/latest.pkl"
    COLLISION_PATH = "/home/antonio/loki/brain/src/experiments/curobo/curobo_loki_one/collision_robot_body.yml"
    INITIAL_JOINT_POSITIONS = [
        9.2086917e-01, -1.7129261e+00, -1.7074385e+00, 2.2253511e+00,  3.7079868e-01, -1.7873198e-03, -6.2069410e-01
    ]
    TARGET_POS_CARRIAGE = np.array([0.41009724, -0.02905086, -0.43246284])
    TARGET_ORIENTATION_CARRIAGE = np.array([2.37053552e-01,  9.69823156e-01,  5.69935205e-02, -6.30584251e-04])  # xyzw

    # create a curobo motion gen instance:
    num_targets = 0
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    RAIL_POSITION = robot_cfg["kinematics"]["lock_joints"]["rail_y_joint"]
    offset = np.array([0.0, 0.0, RAIL_POSITION + FLOOR_TO_RAIL_POSITION])

    def carriage_to_world_taget_pose(target_pos, target_orientation):
        target_pos = target_pos + offset
        target_rot = Rotation.from_quat(target_orientation, scalar_first=False)  # xyzw
        target_rot_inv = target_rot.inv()
        target_quat = target_rot_inv.as_quat(scalar_first=True)  # wxyz
        return np.array(target_pos), np.array(target_quat)

    target_pos, target_quat = carriage_to_world_taget_pose(TARGET_POS_CARRIAGE, TARGET_ORIENTATION_CARRIAGE)

    target = cone.VisualCone(
        "/World/target",
        position=target_pos,
        orientation=target_quat,
        color=np.array([1.0, 0, 0]),
        scale=np.array([0.05, 0.05, 0.05]),
    )

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    articulation_controller = None
    rail_controller = None

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(COLLISION_PATH)
    )
    # translate what is attached to the carriage
    world_cfg_table.cuboid[2].pose[2] += offset[2]
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid)

    trajopt_tsteps = TRAJOPT_STEPS
    trajopt_dt = 1/EXECUTION_FREQ
    optimize_dt = False
    max_attempts = 16
    trim_steps = [1, None]
    interpolation_dt = trajopt_dt
    enable_finetune_trajopt = False
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=16,
        num_graph_seeds=16,
        interpolation_dt=interpolation_dt,
        collision_cache=None, #{"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
        num_ik_seeds=32,
        # grad_trajopt_iters=100,
        # self_collision_check=False,
        # finetune_trajopt_iters=12,
    )
    motion_gen = MotionGen(motion_gen_config)
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=True,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        finetune_attempts=32,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    target_pos_switch_period = 300

    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None

    while simulation_app.is_running():
        my_world.step(render=True)
        
        motion_gen.update_locked_joints(
            {
                "rail_y_joint": RAIL_POSITION,
            },
            robot_cfg,
        )

        def set_initial_joint_positions():
            robot.set_joint_positions(
                np.array([RAIL_POSITION] + INITIAL_JOINT_POSITIONS),
                [0, 1, 2, 3, 4, 5, 6, 7],  # rail_y_joint
            )
        
        def set_joint_positions(positions):
            robot.set_joint_positions(
                np.array(positions),
                [0, 1, 2, 3, 4, 5, 6, 7],  # rail_y_joint
            )

        set_initial_joint_positions()

        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        # if rail_controller is None:
        #     rail_controller = robot.get_articulation_controller()
        #     rail_controller.set_gains(kps=100000.0)
        #     action = ArticulationAction(
        #         np.array([RAIL_POSITION]),
        #         np.array([0.0]),
        #         joint_indices=[0],
        #     )
        #     rail_controller.apply_action(action)
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(len(obstacles.objects))

            motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        # if step_index % target_pos_switch_period == 0:
        # if arm planning frame pos and orientation are the same as target pos and orientation
        if cmd_idx == 0:
            if input("Press ENTER to generate a new trajectory") == "":
                target.set_world_pose(position=target_pos, orientation=target_orientation)
                set_initial_joint_positions()
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
            else:
                exit()
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        robot_static = False
        if (np.max(np.abs(sim_js.velocities)) < 0.5) or args.reactive:
            robot_static = True
        if cmd_plan is None:
            print("Planning...")
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric

            n_manual_retries = 3
            for i in range(n_manual_retries):
                result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
                if result.success:
                    break
                else:
                    print(f"Plan failed for {str(result.status)}, retrying... {i+1}/{n_manual_retries}")
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                if args.reach_partial_pose is not None:
                    reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                print(f"\n\ncmd_plan:\n\n")
                print(cmd_plan)
                print(f"\n\n")
                # print(f"{result.interpolation_dt=}")

                trajectory_dict = {}
                trajectory_dict["joint_names"] = cmd_plan.joint_names
                trajectory_dict["positions"] = cmd_plan.position.cpu().numpy()
                trajectory_dict["velocities"] = cmd_plan.velocity.cpu().numpy()
                trajectory_dict["accelerations"] = cmd_plan.acceleration.cpu().numpy()
                trajectory_dict["jerk"] = cmd_plan.jerk.cpu().numpy()
                with open(TRAJECTORY_OUTPUT_PATH, "wb") as f:
                    pickle.dump(trajectory_dict, f)
                
                print(f"Trajectory saved to {TRAJECTORY_OUTPUT_PATH}")
                print(f"Last robot joint positions (to use as starting configuration for next trajectory): {trajectory_dict['positions'][-1]}")

                cmd_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            target_pose = cube_position
            target_orientation = cube_orientation
        past_pose = cube_position
        past_orientation = cube_orientation
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            # get full dof state

            # set desired joint angles obtained from IK:
            # art_action = ArticulationAction(
            #     cmd_state.position.cpu().numpy(),
            #     cmd_state.velocity.cpu().numpy(),
            #     joint_indices=idx_list,
            # )
            # articulation_controller.apply_action(art_action)
            
            set_joint_positions(cmd_state.position.cpu().numpy())
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
    simulation_app.close()


if __name__ == "__main__":
    main()
