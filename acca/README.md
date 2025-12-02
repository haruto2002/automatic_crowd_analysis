# Automatic Crowd Congestion Analysis System 
![overview](figs/overview.gif)

# Overall Pipeline
<img src="figs/pipeline.png" alt="Pipeline" style="display:block; margin-bottom:30px;" />

# Crowd Risk Analysis
![demo](figs/graph.gif)
<img src="figs/heatmap.png" alt="Heatmap" />

# Setup environment

We utilize Miniconda to create the virtual environment and Python=3.10.  
The required packages are installed by executing `setup.sh`.

```bash
curl -fsSL https://pixi.sh/install.sh | sh

pixi run -e pre-install build-hdf5
pixi run -e pre-install remove-cache
pixi install
```

$HOME/.bash_profileに以下を記入しておくと、次回から自動でロードされる
```bash
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH
```