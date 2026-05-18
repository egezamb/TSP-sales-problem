# Traveling Salesman Problem — Genetic Algorithm Solver

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)

A Python implementation of a **Genetic Algorithm** to solve the classic **Traveling Salesman Problem (TSP)** — finding the shortest possible route that visits each city once and returns to the origin.

Includes a **real-time matplotlib visualization** that plots fitness convergence as the algorithm evolves.

---

## Features

- **Population-based search** — multiple candidate solutions evolved over generations
- **Fitness tracking** — best, worst, and median fitness per generation
- **Real-time plotting** — live matplotlib chart with generation/fitness axes
- **Object-oriented design** — clean separation between Population, RealTimePlot, and algorithm logic
- **Optional background image overlay** on the convergence plot

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| NumPy | Vectorized array operations on populations |
| Matplotlib | Real-time visualization (`TkAgg` backend) |
| Pandas | Data handling |
| Python typing | Type hints for readability |

---

## Running

```bash
git clone https://github.com/egezamb/TSP-sales-problem.git
cd TSP-sales-problem

pip install numpy matplotlib pandas

python qua.py
```

A live plot window will open showing the genetic algorithm's convergence in real time.

---

## What This Project Demonstrates

- **Algorithm design** — implementing a genetic algorithm from scratch
- **Optimization mindset** — population fitness, selection, evolution
- **Scientific Python** — NumPy + Matplotlib for numerical work and visualization
- **Clean OOP** — typed classes, single-responsibility components

---

## Author

**Ege Zambelli** — 3rd-year Software Development student at WSB Merito Wrocław.

- GitHub: [@egezamb](https://github.com/egezamb)
