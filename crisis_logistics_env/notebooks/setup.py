from setuptools import setup


setup(
    name="crisis_logistics_env",
    version="0.1.0",
    description="LogiFlow-RL environment for OpenEnv",
    packages=["crisis_logistics_env", "crisis_logistics_env.server"],
    package_dir={
        "crisis_logistics_env": "..",
        "crisis_logistics_env.server": "../server",
    },
    install_requires=[
        "openenv-core[core]>=0.2.2",
        "numpy>=1.26.0",
        "gymnasium>=0.29.0",
        "openai>=2.0.0",
    ],
)
