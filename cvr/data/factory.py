#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ======================================================================================================================== #
# Project  : Conversion Rate Prediction (CVR)                                                                              #
# Version  : 0.1.0                                                                                                         #
# File     : \make_dataset.py                                                                                              #
# Language : Python 3.7.12                                                                                                 #
# ------------------------------------------------------------------------------------------------------------------------ #
# Author   : John James                                                                                                    #
# Email    : john.james.ai.studio@gmail.com                                                                                #
# URL      : https://github.com/john-james-ai/cvr                                                                          #
# ------------------------------------------------------------------------------------------------------------------------ #
# Created  : Tuesday, December 21st 2021, 7:45:33 pm                                                                       #
# Modified : Wednesday, January 26th 2022, 2:27:41 am                                                                      #
# Modifier : John James (john.james.ai.studio@gmail.com)                                                                   #
# ------------------------------------------------------------------------------------------------------------------------ #
# License  : BSD 3-clause "New" or "Revised" License                                                                       #
# Copyright: (c) 2022 Bryant St. Labs                                                                                      #
# ======================================================================================================================== #
import logging

from cvr.utils.config import CriteoConfig
from cvr.core.workspace import Workspace, WorkspaceManager
from cvr.data.dataset import Dataset, DatasetBuilder
from cvr.core.pipeline import DataPipelineBuilder, DataPipeline
from cvr.data.etl import Extract, TransformETL, LoadDataset

# ------------------------------------------------------------------------------------------------------------------------ #
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------------------------------ #
class DatasetFactor:


def create_workspace(name: str, description: str, current: bool = True, random_state: int = None) -> Workspace:
    """Creates the workspace for the new Dataset object.

    Args:
        name (str): The name for the workspace
        description (str): A description for the workspace
        random_state (int): Seed for pseudo random generator
    """
    wsm = WorkspaceManager()
    if wsm.exists(name):
        workspace = wsm.get_current_workspace(name=name)
    else:
        workspace = wsm.create_workspace(name=name, description=description, current=current, random_state=random_state)
    return workspace


def create_pipeline(
    config: CriteoConfig,
    name: str,
    stage: str,
    force: bool = False,
    chunk_size: int = 20,
    sample_size: int = 10000,
    random_state: int = None,
) -> DataPipeline:
    """Creates the ETL pipeline for the dataset

    Args:
        config (CriteoConfig): The configuration for the data source
        name (str): name of the dataset to be created
        stage (str): stage for the dataset to be created
        force (bool): Whether to force steps already completed
        chunk_size (int): Size of download chunks in Mb
        sample_size(int): Number of random samples to be obtain from the source dataset
        random_state (int): Seed for pseudo random generator
    Returns:
        DataPipeline
    """
    # Tasks of the pipeline
    extract = Extract(config=config, chunk_size=chunk_size, sample_size=sample_size, random_state=random_state)
    transform = TransformETL(value=[-1, "-1"])
    load = LoadDataset()

    # Builder
    builder = DataPipelineBuilder()
    builder.set_name(name).set_stage(stage).set_force(force).set_verbose(True)
    builder.add_task(extract)
    builder.add_task(transform)
    builder.add_task(load)
    builder.build()
    pipeline = builder.pipeline
    return pipeline


def execute_pipeline(pipeline: DataPipeline) -> None:
    pipeline.run()
    _ = pipeline.summary


def main(
    config: CriteoConfig,
    workspace_name: str,
    workspace_description: str,
    dataset_name: str,
    stage: str,
    force: bool = False,
    chunk_size: int = 20,
    sample_size: int = 10000,
    current: bool = True,
    random_state: int = None,
) -> None:

    workspace = create_workspace(
        name=workspace_name, description=workspace_description, current=current, random_state=random_state
    )
    if workspace.dataset_exists(name=dataset_name, stage=stage):
        print("Dataset already exists.")
        dataset = workspace.get_dataset(stage=stage, name=dataset_name)
    else:
        pipeline = create_pipeline(
            config=config,
            name=dataset_name,
            stage=stage,
            force=force,
            chunk_size=chunk_size,
            sample_size=sample_size,
            random_state=random_state,
        )
        dataset = execute_pipeline(pipeline)
    return dataset


if __name__ == "__main__":
    workspace_name = "trieste"
    workspace_description = "Best Neighborhood Cafe in North Beach"
    current = True
    random_state = 601
    config = CriteoConfig()
    dataset_name = "french_roast"
    stage = "beat"
    force = False
    chunk_size = 50
    sample_size = 10000
    dataet = main(
        config=config,
        workspace_name=workspace_name,
        workspace_description=workspace_description,
        dataset_name=dataset_name,
        stage=stage,
        force=force,
        chunk_size=chunk_size,
        sample_size=sample_size,
        current=current,
        random_state=random_state,
    )
