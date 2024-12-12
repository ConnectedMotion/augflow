<p align="center">
    <a href="https://www.connectedmotion.io/">
        <img src="https://i.ibb.co/FBTFwtG/augflow-high-resolution-logo-transparent-5.png" style="margin-left: -10px;" border="0">
    </a>
</p>




<!-- <h1 align="center">AugFlow</h1> -->



<p align="center">
	<img src="https://img.shields.io/github/license/eli64s/readme-ai?style=default&logo=opensourceinitiative&logoColor=white&color=4c3c84" alt="license">
	<a href="https://"><img src="https://img.shields.io/badge/Last_Commit-November-4c3c84?logo=github" alt="Last Commit - November"></a>
    <a href="https://"><img src="https://img.shields.io/badge/Python-100%25-4c3c84?logo=Python" alt="Python - 100%"></a>
    <a href="https://colab.research.google.com/drive/1kEJ6SK68jwO3NzjooWa1esZgOJ8Rge2L?usp=sharing"><img src="https://img.shields.io/badge/Colab-Notebook-4c3c84?logo=googlecolab" alt="Colab - Notebook"></a>

    
    
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>



<p align="center">
  <b>Unlocking Infinite Possibilities in Computer Vision Augmentation</b><br>
  <sub>🔧 Configurable • ⚙️ Controllable • 📄 Format-Agnostic</sub>

</p>


<br>




<div style="margin-left: 40px;">

**AugFlow** is an advanced open-source augmentation library designed to elevate your computer vision projects. Whether you're a researcher pushing innovation or a practitioner aiming for peak performance, AugFlow enables the creation of highly customizable augmentation pipelines that enhance dataset diversity and improve model generalization. With fully configurable and controllable processes, AugFlow offers unmatched flexibility for precise fine-tuning, optimizing both performance and robustness. 🚀


To explore how AugFlow builds diverse augmentation pipelines, refer to the **Flow** diagram below.

</div>

![Flow Diagram](/assets/flow.png)




---
## 👾 Features
<div style="margin-left: 40px;">

  **Discover What Makes AugFlow Unique and So Powerful 🌟**

- **🌊 Innovative Flow Concept**
    AugFlow introduces the entirely novel concept of flows, a groundbreaking approach not yet implemented in any existing theory or solutions. Our initial evaluations demonstrate the impact and advantages this unique methodology brings to data augmentation.

- **🛠️ Configurable Parameters**: Every augmentation is fully configurable through simple and intuitive parameters.

- **🖍️ Comprehensive Polygon Clipping Control**
    Unlike existing open and closed-source solutions that inadequately manage polygon clipping, AugFlow offers comprehensive, per-class control over clipped polygons, enhancing dataset diversity especially for those with high intra-class variability.

- **✂️ Enhanced CutOut Functionality**
    AugFlow's CutOut feature provides superior control, enabling precise targeting of large or small polygons along with customizable parameters. This flexibility ensures more effective and tailored data augmentation strategies.

- **🧩 Advanced Mosaic Augmentation**
    AugFlow's Mosaic augmentation offers exceptional control with dynamic grid configurations, including larger grids for high-resolution images. This adaptability significantly improves detection accuracy for small objects within complex scenes.

- **🔄 Precise Transformation Controls**
    AugFlow enhances fundamental augmentations like Crop, Rotate, Shear, and Translate, allowing precise customization to better suit diverse dataset requirements and improve model robustness.

- **🎯 Targeted Category Focusing**
    AugFlow enables targeted augmentation by focusing on images containing specific categories while excluding others, effectively balancing datasets. Future enhancements aim to handle edge cases per class using advanced unsupervised techniques.

- **📄 Format-Agnostic**: Supports multiple annotation formats including **COCO** and **YOLO** seamlessly, for both detection and segmentation.

- **📊 Logging & Visualization**: Track augmentation processes with detailed logs and visualize annotations to ensure quality.

- **🔧 Extensible Architecture**: Designed to easily incorporate new augmentations and extend existing functionalities.

</div>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.6 or higher
- **Package Manager:** [pip](https://pip.pypa.io/en/stable/) installed
- **Git:** Installed for cloning the repository and contributing ([Download Git](https://git-scm.com/downloads))



### ⚙️ Installation

🛠️ Setting Up a Virtual Environment (Recommended):

1. **Create a Virtual Environment:**

    ```bash
    python3 -m venv augflow-env
    ```

2. **Activate the Virtual Environment:**

    - **Windows:**

      ```bash
      augflow-env\Scripts\activate
      ```

    - **macOS and Linux:**

      ```bash
      source augflow-env/bin/activate
      ```


After creating your virtual env you can **Install AugFlow** using one of the following methods:

1. **Build from source:**


    - Clone the augflow repository:
        ```bash
        git clone https://github.com/ConnectedMotion/augflow.git
        ```

    - Navigate to the project directory:
        ```bash
        cd augflow
        ```


    - Install the project dependencies:

        ```bash
        pip install -e .
        ```

2. **Or. Install directly using &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)**

 ```bash
    pip install augflow
 ```

### 🎯 Quick Start Example

The following example demonstrates how to use AugFlow's Pipeline to apply Augmentations to coco 128 dataset.

**First, download the COCO128 dataset from the following [Google Drive link](https://drive.google.com/file/d/1xiH3vwa2ZpgmytmuVA00M6FGA3DXmGUH/view?usp=sharing), extract it, and run the example.**

After running your experiment, you will get detailed logs about the flow, saved in the **augflow_pipeline.log** file.

  ```bash
    


import os
import logging
from augflow.pipeline import Pipeline
import augflow.utils.configs as config
from augflow.utils.configs import example_custom_rotate_config

logging.basicConfig(level=logging.INFO)

experiment_id = 'exp1'


pipe = Pipeline()


pipe.task(
    format='yolo',
    dataset_path='dataset_path_contains_data_dot_yaml_images_labels'
)

pipe.fuse(
    source_id='root',
    type='rotate',
    id='aug_rotate',
    config=example_custom_rotate_config,
    merge=True
)


pipe.out(
    format='yolo',
    output_path=f'output_path/{experiment_id}',
    ignore_masks=False,
    visualize_annotations=True
)



  ```

### 🔬 Unit Testing
Run the test suite using **PyTest**:

```sh
pytest tests/ -v
```


---

## 📌 Project Roadmap

- [✅] **Milestone 1**: Implement more controllable, configurable, and intelligent augmentation flows (Done)

- [❌] **Milestone 2**: Develop Detailed Documentation. 🕒

- [❌] **Milestone 3**: Add Comprehensive Control Over Individual Augmentations. 🕒

- [❌] **Milestone 4**: Provide predefined augmentation presets based on the dataset's statistical distribution. 🕒

- [❌] **Milestone 5**: Support Key Points Annotations. 🕒

- [❌] **Milestone 6**: Enhance Augmentation Focus Based on Object Features. 🕒

- [❌] **Milestone 7**: Support 3D Augmentations. 🕒

- [❌] **Milestone 8**: Develop an Interactive GUI for Augmentation Setup. 🕒



 

    

---


## 🔰 Contributing

- **💬 [Join the Discussions](https://github.com/ConnectedMotion/augflow/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/ConnectedMotion/augflow/issues)**: Submit bugs found or log feature requests for the `augflow` project.
- **💡 [Submit Pull Requests](https://github.com/ConnectedMotion/augflow/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.




1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
```sh
git clone https://github.com/your-username/augflow.git

```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
```sh
git checkout -b new-feature-x
```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
```sh
git commit -m 'Implemented new feature x.'
```
6. **Push to github**: Push the changes to your forked repository.
```sh
git push origin new-feature-x
```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!

---

**🎗 License**

This project is licensed under the **MIT** License. For more details, refer to the LICENSE file.

---

**📬 Contact and Support**

For any inquiries or support, feel free to reach out via [**Issues**](https://github.com/ConnectedMotion/augflow/issues) or join the [**Discussions**](https://github.com/ConnectedMotion/augflow/discussions).
