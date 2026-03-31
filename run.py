#!/usr/bin/env python3
"""
项目主入口
用法:
  python run.py train          # 运行完整训练Pipeline
  python run.py serve          # 启动API服务
  python run.py evaluate       # 运行评估
  python run.py test           # 运行测试
"""
import sys
import click
from loguru import logger


@click.group()
def cli():
    """多模态决策强化学习 - 智能内容生成与推荐系统"""
    pass


@cli.command()
@click.option("--config", default="configs/base_config.yaml", help="配置文件路径")
@click.option("--stage", default="all", type=click.Choice(["all", "sft", "rm", "dpo", "ppo"]))
def train(config, stage):
    """运行训练Pipeline"""
    from src.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config)

    if stage == "all":
        results = pipeline.run_full_pipeline()
    elif stage == "sft":
        results = pipeline.stage_sft()
    elif stage == "rm":
        results = pipeline.stage_reward_model()
    elif stage == "dpo":
        results = pipeline.stage_dpo()
    elif stage == "ppo":
        results = pipeline.stage_ppo()

    logger.info(f"✅ 训练完成: {stage}")


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
@click.option("--workers", default=2, type=int)
def serve(host, port, workers):
    """启动API服务"""
    import uvicorn
    logger.info(f"🚀 启动API服务: {host}:{port}")
    uvicorn.run(
        "src.api.server:app",
        host=host, port=port,
        workers=workers, reload=False,
    )


@cli.command()
@click.option("--config", default="configs/base_config.yaml")
@click.option("--output", default="outputs/eval_report.json")
def evaluate(config, output):
    """运行评估"""
    import torch
    from src.evaluation.metrics import EvaluationSuite

    suite = EvaluationSuite(device="cpu")

    # 模拟评估数据
    chosen = torch.randn(100) + 0.5
    rejected = torch.randn(100) - 0.5
    baseline = torch.randn(100)
    optimized = torch.randn(100) + 1.0

    rl_results = suite.evaluate_rl(chosen, rejected, baseline, optimized)

    # 业务指标
    from src.evaluation.metrics import BusinessMetrics
    scores = torch.randn(100)
    ctr = BusinessMetrics.simulate_ctr(scores)
    engagement = BusinessMetrics.simulate_engagement(scores)

    results = {
        "rl_metrics": rl_results,
        "business_metrics": {"ctr": ctr, **engagement},
    }

    suite.generate_report(results, output)
    logger.info(f"📊 评估完成，报告: {output}")


@cli.command()
def test():
    """运行测试"""
    import pytest
    sys.exit(pytest.main(["-v", "tests/", "--tb=short"]))


if __name__ == "__main__":
    cli()
