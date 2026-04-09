"""
分阶段运行训练Pipeline + 推理演示
自动处理CPU环境和缺少数据的情况
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from loguru import logger

logger.remove()
# Windows 终端 GBK 编码下无法输出 emoji，使用 utf-8 安全输出
import io
_sink = sys.stdout
try:
    if hasattr(sys.stdout, "buffer"):
        _sink = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
logger.add(_sink, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


def run_training():
    """执行完整训练Pipeline"""
    from src.training.pipeline import TrainingPipeline

    logger.info("=" * 60)
    logger.info("🚀 开始电商营销文案系统 - 训练Pipeline")
    logger.info("=" * 60)

    start_time = time.time()

    pipeline = TrainingPipeline("configs/base_config.yaml")

    # Stage 1: SFT
    logger.info("\n" + "=" * 50)
    logger.info("📋 Stage 1/4: SFT 监督微调")
    logger.info("=" * 50)
    t1 = time.time()
    sft_result = pipeline.stage_sft()
    logger.info(f"✅ SFT完成 (耗时: {time.time()-t1:.1f}s)")

    # Stage 2: RM
    logger.info("\n" + "=" * 50)
    logger.info("📋 Stage 2/4: 奖励模型训练")
    logger.info("=" * 50)
    t2 = time.time()
    rm_result = pipeline.stage_reward_model()
    rm_final = rm_result.get("final_metrics", {})
    logger.info(f"✅ 奖励模型完成 (耗时: {time.time()-t2:.1f}s)")
    if rm_final:
        logger.info(f"   最终指标: loss={rm_final.get('loss', 'N/A'):.4f}, acc={rm_final.get('accuracy', 'N/A'):.4f}")

    # Stage 3: DPO
    logger.info("\n" + "=" * 50)
    logger.info("📋 Stage 3/4: DPO 偏好对齐")
    logger.info("=" * 50)
    t3 = time.time()
    dpo_result = pipeline.stage_dpo()
    dpo_final = dpo_result.get("final_metrics", {})
    logger.info(f"✅ DPO完成 (耗时: {time.time()-t3:.1f}s)")
    if dpo_final:
        logger.info(f"   最终指标: loss={dpo_final.get('loss', 'N/A'):.4f}, acc={dpo_final.get('accuracy', 'N/A'):.4f}")

    # Stage 4: PPO
    logger.info("\n" + "=" * 50)
    logger.info("📋 Stage 4/4: PPO 决策优化")
    logger.info("=" * 50)
    t4 = time.time()
    ppo_result = pipeline.stage_ppo()
    ppo_final = ppo_result.get("final_metrics", {})
    logger.info(f"✅ PPO完成 (耗时: {time.time()-t4:.1f}s)")
    if ppo_final:
        logger.info(
            f"   最终指标: policy_loss={ppo_final.get('policy_loss', 'N/A'):.4f}, "
            f"value_loss={ppo_final.get('value_loss', 'N/A'):.4f}, "
            f"kl={ppo_final.get('kl_div', 'N/A'):.4f}"
        )

    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 全流程训练完成! 总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info("=" * 60)

    return {
        "sft": sft_result,
        "reward_model": rm_result,
        "dpo": dpo_result,
        "ppo": ppo_result,
        "total_time": total_time,
    }


if __name__ == "__main__":
    results = run_training()
