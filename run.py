#!/usr/bin/env python3
"""
项目主入口 — 电商智能营销文案系统
用法:
  python run.py train          # 运行完整训练Pipeline（电商数据）
  python run.py serve          # 启动API服务
  python run.py evaluate       # 运行评估
  python run.py demo           # 运行电商文案生成演示
  python run.py test           # 运行测试
"""
import sys
import click
from loguru import logger


@click.group()
def cli():
    """电商智能营销文案系统 — 多模态决策强化学习驱动"""
    pass


@cli.command()
@click.option("--config", default="configs/base_config.yaml", help="配置文件路径")
@click.option("--stage", default="all", type=click.Choice(["all", "sft", "rm", "dpo", "ppo"]))
def train(config, stage):
    """运行训练Pipeline（支持电商数据）"""
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
    """启动API服务（含电商文案生成接口）"""
    import uvicorn
    logger.info(f"🚀 启动电商营销文案API服务: {host}:{port}")
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

    # 评估数据
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
@click.option("--product", default="水光精华液", help="商品名称")
@click.option("--description", default="深层补水，24小时持续保湿", help="商品描述")
@click.option("--category", default="美妆", help="商品类目")
@click.option("--price", default=168.0, type=float, help="价格")
@click.option("--styles", default=None, help="风格列表(逗号分隔)，默认全部")
def demo(product, description, category, price, styles):
    """运行电商营销文案生成演示"""
    logger.info("=" * 60)
    logger.info("🛒 电商营销文案生成演示")
    logger.info("=" * 60)

    from src.generation.copy_engine import CopyGenerationEngine

    engine = CopyGenerationEngine(use_model=False)

    style_list = styles.split(",") if styles else None
    tags = []

    results = engine.generate_multi_style_copies(
        product_title=product,
        product_description=description,
        category=category,
        tags=tags,
        price=price,
        styles=style_list,
        num_candidates=3,
    )

    logger.info(f"\n📦 商品: {product}")
    logger.info(f"📝 描述: {description}")
    logger.info(f"🏷️ 类目: {category} | 💰 价格: ¥{price:.0f}")
    logger.info("-" * 60)

    for style, copies in results.items():
        logger.info(f"\n🎨 风格: {style}")
        for i, copy in enumerate(copies):
            logger.info(f"  [{i+1}] (分数: {copy['score']:.3f}) {copy['content'][:120]}...")

    logger.info("\n" + "=" * 60)
    logger.info("✅ 演示完成！")

    # 文案质量评估演示
    from src.models.copy_generator import CopyQualityEvaluator
    evaluator = CopyQualityEvaluator()

    logger.info("\n📊 文案质量评估:")
    for style, copies in results.items():
        if copies:
            best = copies[0]
            scores = evaluator.evaluate_copy(best["content"], style)
            logger.info(
                f"  {style}: overall={scores['overall']:.3f} "
                f"(length={scores['length_score']:.2f}, "
                f"readability={scores['readability_score']:.2f}, "
                f"style_match={scores['style_match_score']:.2f})"
            )


@cli.command()
def test():
    """运行测试"""
    import pytest
    sys.exit(pytest.main(["-v", "tests/", "--tb=short"]))


if __name__ == "__main__":
    cli()
