# Multi-scale Laplacian-based FMM for elastic shape mapping

This repository provides a basic Python implementation for shape analysis and comparison using multiscale Laplacian coordinates and the Fast March Method (FMM). It is not optimised for performance, as it is intended for illustrative purposes.


If you use this code, please cite our conference paper (journal extension comming soon):


> **[1]** I. Cuiral-Zueco and G. López-Nicolás, "Multi-scale Laplacian-based FMM for shape control," *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, Prague, Czech Republic, 2021, pp. 3792-3797, [doi: 10.1109/IROS51168.2021.9636857](https://doi.org/10.1109/IROS51168.2021.9636857).

**Input Shapes:**

| Shape 1 | Shape 2 |
|---|---|
| <img src="https://github.com/user-attachments/assets/8f7a2a69-5c3f-49c3-acf3-b36fa137dfee" width="200"> | <img src="https://github.com/user-attachments/assets/1ccc8b10-bba9-47b6-a861-a6e88c353cd8" width="200"> |

**FMM Optimization:**

The FMM is used to compute the optimal elastic map between both shapes:

| FMM optimisation Result |
|---|
| <img src="https://github.com/user-attachments/assets/bbfeba19-42f4-4d22-8b41-4ed6ff288d6e" width="200"> |

**Elastic Mapping:**

Mapping results:

| Elastic Mapping Result |
|---|
| <img src="https://github.com/user-attachments/assets/15d5fbea-67f3-454a-9875-dd0b3ae3e4cc" width="200"> |

**Dependencies:**

- numpy
- opencv-python
- scipy
- matplotlib
- scikit-image

**Functionality:**

- Contour extraction
- Contour sampling
- Laplacian coordinate computation
- Tangent/normal projection of Laplacian vectors
- F-surface construction
- Shortest path finding
- Contour interpolation and alignment

**Usage:**

1. Install dependencies: `pip install numpy opencv-python scipy matplotlib scikit-image`
2. Modify image paths in the `if __name__ == "__main__":` block (optional).
3. Run the script: `python shape_analysis.py`
---
# Citation

The related paper can be cited as:

> **[1]** I. Cuiral-Zueco and G. López-Nicolás, "Multi-scale Laplacian-based FMM for shape control," *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, Prague, Czech Republic, 2021, pp. 3792-3797, [doi: 10.1109/IROS51168.2021.9636857](https://doi.org/10.1109/IROS51168.2021.9636857).

- [Ignacio Cuiral-Zueco](https://nachocz.github.io/icz-cv/)
- [Gonzalo López-Nicolás](http://webdiis.unizar.es/~glopez)

# Acknowledgements

This work was supported through Project REMAIN S1/1.1/E0111 (Interreg Sudoe Programme, ERDF), Project PID2021-124137OB-I00, and Project TED2021-130224B-I00 funded in part by MCIN/AEI/10.13039/501100011033, in part by the ERDF A way of making Europe, and in part by the European Union NextGenerationEU/PRTR.

<img src="https://github.com/user-attachments/assets/44c6e2a3-e281-4e59-a594-fcf72769a4b9" height="100px">
<img src="https://github.com/user-attachments/assets/ed25931c-3dc5-4db3-8c63-13883d3ef100" height="100px">
