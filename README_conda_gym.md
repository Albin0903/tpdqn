## Installation environnement virtuel avec conda

[Conda](https://docs.conda.io/en/latest/) est un système de gestion de package permettant l'installation de multiples versions de logiciels au travers d'un mécanisme d'**environnements virtuels**. Vous pouvez ainsi isoler vos différents projets Python dans des environnements virtuels différents. Chaque environnement virtuel utilisera la version souhaitée de Python et des packages associés pour votre projet.

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.



### 1. Installation d'Anaconda ou de miniconda

Vous pouvez au choix installer [Anaconda](https://www.anaconda.com/products/individual), qui contient le gestionnaire de paquet Conda, plus les bibliothèques scientifiques, un environnement de développement … ou uniquement le gestionnaire de paquet Conda appelé [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Pour Anaconda, suivez les consignes sur leur site. Pour Miniconda, suivez les consignes ci-dessous:

> [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a free minimal installer for conda. 

**Télécharger** la denrière version de`miniconda` correspondant à votre système.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Installer** [miniconda](http://conda.pydata.org/miniconda.html) sur votre machine:

- **Linux:** https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
- **Mac:** https://conda.io/projects/conda/en/latest/user-guide/install/macos.html#install-macos-silent
- **Windows:** https://conda.io/projects/conda/en/latest/user-guide/install/windows.html

### 2. Creation et activation d'un environnement

Ci-dessous:
* pour Mac et Linux, les commandes sont à faire dans un terminal classique. 
* Pour Windows, il faut utiliser **Anaconda prompt** et pas un terminal de commande classique (taper "Anaconda Prompt" dans la barre de recherche Windows). 

1. Créer (et activer) un nouvel environnement, appelé `tpdeeprl2024`:

```
conda create --name tpdeeprl2024 python=3.10
conda activate tpdeeprl2024
```

A ce niveau, votre ligne de commande doit ressembler à : `(tpdeeprl2024) <User>: `. 

`(tpdeeprl2024)` indique que l'environnement créé est actif, et vous pouvez maintenant installer des packages dans l'environnement.


2. Installation de PyTorch et torchvision:

-  Sur __Windows__: 
```
conda install pytorch==2.0.1  torchvision -c pytorch
conda install m2-base
```
- Sur __Mac__:
```
conda install pytorch==2.0.1  torchvision -c pytorch

```
- Sur __Linux__ : 
```
conda install pytorch=2.0.1 -c pytorch 
pip install torchvision
```
4. Git

Dans la suite il est supposé que `git` est installé sur votre machine. Si ce n'est pas le cas, vous pouvez utiliser `conda`pour l'installer:
```
conda install git
```

5. Cloner le dépôt créé *via* githubclassroom et aller dans le dossier du dépôt:
```
git clone https://github.com/X.git
cd X
```

6. Installation des packages spécifiés dans le fichier *requirements.txt*.
```
pip install -r requirements.txt
```
7. Installation de gymnasium
-  Sur __Windows__:
```
pip install swig
```
Ensuite aller sur https://visualstudio.microsoft.com/visual-cpp-build-tools/
 -> cliquer sur" Télécharger Build tools", puis lancer l'installer installé. Lors du choix, sélectionner "Desktop Development with C++" et garder les autres éléments facultatifs cochés par défaut.
Une fois installé:
```
pip install gymnasium[box2d]
```

- Sur __Linux__: 
```
conda install conda-forge::gymnasium
conda install conda-forge::gymnasium-box2d
```
- Sur __Mac__:
```
conda install -c conda-forge gymnasium
conda install swig
conda install -c conda-forge gym-box2d
```


8. Vous pouvez maintenant  commencer à compléter le notebook `TPDQN.ipynb` pour faire votre TP, soit avec `jupyter-lab`, ou (conseillé) avec l'[extension Jupyter de VisualStudio](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) qui permet d'utiliser le debugger.


## Sources

- si besoin, utiliser ([google colab](https://colab.research.google.com/?hl=fr)), version cloud de jupyter notebook qui  permet d'accéder gratuitement à des ressources informatiques, dont des GPU (limité).
- Un tutoriel sur les [Jupyter notebook](https://python.sdv.univ-paris-diderot.fr/18_jupyter/)
- Vous pouvez lister les environnements conda installés :
```
conda env list
```
- Vous pouvez lister les packages installés dans l'environnement actif :
```
conda list
```

