import asyncio

from models import CrisisLogisticsAction
from server.crisis_logistics_env_environment import (
    CrisisLogisticsEnvironment,
    choose_balancing_action,
)


async def test_run():
    print("--- BOOTING LOGIFLOW-RL SIMULATOR ---")
    env = CrisisLogisticsEnvironment()

    print("\n[INIT] Resetting Environment...")
    obs = env.reset(task_id="medium")
    print(f"Task: {obs.task_id} ({obs.difficulty})")
    print(f"Objective: {obs.objective}")
    print(f"Hub Loads: {obs.hub_loads}")
    print(f"Incoming Load: {obs.incoming_load}")
    print(f"Event Type: {obs.event_label}")

    suggested_hub = choose_balancing_action(obs)
    print(f"\n[POLICY] Baseline sends shipment to hub {suggested_hub}...")
    obs = env.step(CrisisLogisticsAction(target_hub=suggested_hub))

    print(f"Updated Loads: {obs.hub_loads}")
    print(f"Reward Received: {obs.reward}")
    print(f"Current Score: {obs.cumulative_score}")
    print(f"Next Incoming Load: {obs.incoming_load}")
    print(f"Message: {obs.message}")


if __name__ == "__main__":
    asyncio.run(test_run())
