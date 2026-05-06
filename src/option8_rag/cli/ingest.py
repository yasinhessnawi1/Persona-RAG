"""Ingest a corpus.

Two backends are wired by ``corpus.name``:

- ``beir_nq``: load BEIR/NQ from HuggingFace; persist nothing extra (the HF
  cache already stores the dataset).
- ``uia_ikt``: discover URLs from the sitemap, then crawl politely.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from option8_rag.cli._common import resolve_paths, set_seed, setup_logging
from option8_rag.ingest.beir import BeirLoader
from option8_rag.ingest.uia import CrawlConfig, UiaCrawler


@hydra.main(
    config_path="../config",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    set_seed(int(cfg.seed))
    paths = resolve_paths(cfg)

    name = str(cfg.corpus.name)
    logger.info("ingest corpus={name}", name=name)

    if name == "beir_nq":
        _ingest_beir(cfg=cfg, paths=paths)
    elif name == "uia_ikt":
        _ingest_uia(cfg=cfg, paths=paths)
    else:
        raise ValueError(f"unknown corpus {name!r}")


def _ingest_beir(*, cfg: DictConfig, paths: dict[str, Path]) -> None:
    loader = BeirLoader(
        corpus_dataset=cfg.corpus.hf.corpus_dataset,
        corpus_config=cfg.corpus.hf.corpus_config,
        corpus_split=cfg.corpus.hf.corpus_split,
        queries_dataset=cfg.corpus.hf.queries_dataset,
        queries_config=cfg.corpus.hf.queries_config,
        queries_split=cfg.corpus.hf.queries_split,
        qrels_dataset=cfg.corpus.hf.qrels_dataset,
        qrels_split=cfg.corpus.hf.qrels_split,
        cache_dir=paths["cache_root"],
    )
    full_corpus = bool(cfg.corpus.full_corpus)
    smoke = bool(cfg.run.smoke)
    if smoke:
        max_corpus = int(cfg.run.smoke_max_chunks)
        logger.warning("smoke mode: capping BEIR corpus to {n}", n=max_corpus)
    elif full_corpus:
        max_corpus = None
    else:
        max_corpus = int(cfg.corpus.subsample_size)

    split = loader.load(max_corpus=max_corpus)
    logger.info(
        "BEIR ingest: {n_corpus} docs, {n_queries} queries, {n_qrels} judged",
        n_corpus=len(split.corpus),
        n_queries=len(split.queries),
        n_qrels=len(split.qrels),
    )


def _ingest_uia(*, cfg: DictConfig, paths: dict[str, Path]) -> None:
    out_dir = paths["raw_root"] / "uia_ikt"
    crawl_cfg = CrawlConfig(
        sitemap_url=str(cfg.corpus.crawl.sitemap_url),
        url_regex=str(cfg.corpus.crawl.url_regex),
        extra_urls=tuple(str(u) for u in cfg.corpus.crawl.extra_urls or []),
        user_agent=str(cfg.corpus.crawl.user_agent),
        delay_seconds=float(cfg.corpus.crawl.delay_seconds),
        jitter_seconds=float(cfg.corpus.crawl.jitter_seconds),
        max_pages=int(cfg.corpus.crawl.max_pages),
        obey_robots=bool(cfg.corpus.crawl.obey_robots),
        latest_year_only=bool(cfg.corpus.crawl.get("latest_year_only", False)),
    )
    crawler = UiaCrawler(config=crawl_cfg, out_dir=out_dir)

    urls = crawler.discover_urls()
    if cfg.run.smoke:
        cap = int(cfg.run.smoke_max_chunks)
        urls = urls[:cap]
        logger.warning("smoke mode: capping UiA crawl to {n} pages", n=cap)
    documents = crawler.crawl(urls)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "n_urls_discovered": len(urls),
                "n_pages_persisted": len(documents),
                "out_dir": str(out_dir),
            },
            indent=2,
        ),
    )
    logger.info("UiA ingest summary written to {p}", p=str(summary_path))


if __name__ == "__main__":
    main()
