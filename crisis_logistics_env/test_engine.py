import asyncio

from server.crisis_logistics_env_environment import (
    CrisisLogisticsEnvironment,
    choose_network_action,
    choose_resilient_action,
)


async def test_run():
    print("--- BOOTING LOGIFLOW-RL SIMULATOR ---")
    env = CrisisLogisticsEnvironment()

    print("\n[INIT] Resetting Environment...")
    obs = env.reset(task_id="medium")
    print(f"Task: {obs.task_id} ({obs.difficulty})")
    print(f"Objective: {obs.objective}")
    print(f"Tier Loads: {obs.hub_loads}")
    print(f"Node Count: {len(obs.node_loads)}")
    print(f"Visible Nodes: {obs.visible_node_ids}")
    print(f"Incoming Load: {obs.incoming_load}")
    print(f"Event Type: {obs.event_label}")

    action = choose_network_action(obs)
    print(
        f"\n[POLICY] Baseline routes {action.shipment_volume} units "
        f"from node {action.source_node} to node {action.dest_node}..."
    )
    obs = env.step(action)

    print(f"Updated Tier Loads: {obs.hub_loads}")
    print(f"In Transit: {len(obs.in_transit_shipments)}")
    print(f"Reward Breakdown: {obs.reward_breakdown}")
    print(f"Reward Received: {obs.reward}")
    print(f"Current Score: {obs.cumulative_score}")
    print(f"Dynamic Pressure: {obs.dynamic_pressure}")
    print(f"Priority Target: {obs.priority_target_name} (node {obs.priority_target_node})")
    print(f"Priority Service Rate: {obs.priority_service_rate}")
    print(f"Next Incoming Load: {obs.incoming_load}")
    print(f"Message: {obs.message}")

    resilient_action = choose_resilient_action(obs)
    print(
        f"\n[RESILIENT] Routes {resilient_action.shipment_volume} units "
        f"from node {resilient_action.source_node} to node {resilient_action.dest_node}..."
    )
    obs = env.step(resilient_action)
    print(f"Reward Breakdown: {obs.reward_breakdown}")
    print(f"Reward Received: {obs.reward}")
    print(f"Current Score: {obs.cumulative_score}")


if __name__ == "__main__":
    asyncio.run(test_run())
