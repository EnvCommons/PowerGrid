import json
import asyncio
import os
import datetime

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    or_client = AsyncOpenReward(base_url="http://localhost:8082")
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "powergridenvironment"
    SPLIT = "train"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    environment = or_client.environments.get(name=ENV_NAME)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")
    print(f"Tools: {[t['name'] for t in tools]}")

    for task in tasks[:1]:  # Test first task
        task_id = task.task_spec["id"]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"trajectory_{task_id}_{timestamp}.jsonl"

        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Logging to: {log_file}")
        print(f"{'='*60}\n")

        rollout = or_client.rollout.create(
            run_name=ENV_NAME + "_test",
            rollout_name="test_run",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec,
        )

        async with environment.session(
            task=task, secrets={"openai_api_key": OPENAI_API_KEY}
        ) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            # Log initial prompt
            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "type": "prompt",
                    "task_id": task_id,
                    "content_preview": prompt[0].text[:500],
                    "timestamp": datetime.datetime.now().isoformat(),
                }) + "\n")

            rollout.log_openai_response(message=input_list[0], is_finished=finished)

            step = 0
            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )

                rollout.log_openai_response(response.output[-1])
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        tool_result = await session.call_tool(
                            item.name, json.loads(str(item.arguments))
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished
                        metadata = tool_result.metadata if hasattr(tool_result, 'metadata') else {}

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })
                        rollout.log_openai_response(
                            input_list[-1],
                            reward=reward,
                            is_finished=finished,
                        )

                        # JSONL trajectory logging
                        log_entry = {
                            "type": "tool_call",
                            "step": step,
                            "tool": item.name,
                            "arguments": json.loads(str(item.arguments)),
                            "reward": reward,
                            "finished": finished,
                            "metadata": metadata,
                            "timestamp": datetime.datetime.now().isoformat(),
                        }
                        with open(log_file, "a") as f:
                            f.write(json.dumps(log_entry) + "\n")

                        # Console output
                        freq = metadata.get("frequency_hz", "?")
                        demand = metadata.get("total_demand_mw", "?")
                        gen = metadata.get("total_generation_mw", "?")
                        shed = metadata.get("total_load_shed_mw", "?")
                        cum_r = metadata.get("cumulative_reward", "?")

                        print(
                            f"Step {step:3d} | {item.name:<25s} | "
                            f"R={reward:+.3f} | CumR={cum_r} | "
                            f"f={freq} Hz | D={demand} MW | G={gen} MW | Shed={shed} MW"
                        )
                        step += 1

                        if tool_result.finished:
                            finished = True
                            reason = metadata.get("reason", "unknown")
                            print(f"\nFINISHED: {reason}")
                            print(f"Final cumulative reward: {cum_r}")

                            # Log final summary
                            with open(log_file, "a") as f:
                                f.write(json.dumps({
                                    "type": "summary",
                                    "total_steps": step,
                                    "final_reward": cum_r,
                                    "reason": reason,
                                    "finished": True,
                                    "timestamp": datetime.datetime.now().isoformat(),
                                }) + "\n")
                            break

        print(f"\nTrajectory saved to: {log_file}")
        print(f"Total tool calls: {step}")


if __name__ == "__main__":
    asyncio.run(main())
