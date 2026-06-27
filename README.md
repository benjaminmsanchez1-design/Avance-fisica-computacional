# Física Computacional - FIS 205

Este repositorio contiene los códigos en Python, simulaciones numéricas e informes desarrollados para la asignatura de Física Computacional, correspondiente a la Licenciatura en Física de la Universidad Técnica Federico Santa María (USM).

---

## 📂 Estructura del Repositorio

El repositorio está organizado para separar las tareas regulares de los distintos avances del proyecto semestral:

* **[`/Tarea 1`](./Tarea%201):** 
* **[`/Tarea 2`](./Tarea%202):** 
* **Proyecto Semestral: Propagación de Ondas Sísmicas 2D**
  * **[`/avance1`](./avance1):** Planteamiento inicial del modelo físico y la fuente.
  * **[`/avance 2`](./avance%202):** Implementación del esquema numérico, diferencias finitas y simulación base.
  * **[`/avance 3`](./avance%203):** Mejoras en las condiciones de frontera, preparación de modelos complejos y redacción del informe.

---

## 🌋 Proyecto Semestral: Propagación de Ondas Sísmicas 2D

**Descripción:** Simulación numérica de la propagación de ondas sísmicas acústicas en medios geológicos heterogéneos por capas, basada en el trabajo clásico de *Virieux (1986)*. El proyecto implementa un solver de la ecuación de onda 2D mediante diferencias finitas de **4º orden en espacio** y **2º orden en tiempo**, excitado por un pulso de **Ricker** (25 Hz), con condiciones de frontera absorbentes tipo *sponge layer*. El objetivo final es generar sismogramas sintéticos que sirvan de base para una futura inversión tomográfica.

### 📈 Estado de Avance 

A continuación se detalla el progreso de los objetivos computacionales y físicos, reflejados en las distintas carpetas de avance del repositorio:

#### ✅ Fase 1: Modelo Físico y Fuente (`avance1`)
* Formulación de la ecuación de onda acústica 2D en medio heterogéneo.
* Implementación del pulso de Ricker con frecuencia central de 25 Hz (`fuente.py`).
* Verificación del contenido espectral de banda limitada (FFT del pulso).
* Corrección del retardo temporal `t0 = 1/fp` para garantizar causalidad y evitar truncamiento.

#### ✅ Fase 2: Esquema Numérico y Simulación (`avance 2`)
* Discretización del Laplaciano con estarcido (*stencil*) de 4º orden, vectorizado con `numpy`.
* Integración temporal explícita tipo *leapfrog* de 2º orden.
* Verificación automática del criterio de estabilidad **CFL** (Courant = 0.40).
* Construcción y lectura del modelo de velocidad por capas en formato `.npy` (`modelo.py`).
* Implementación de capa absorbente (*sponge layer*) para reducir reflexiones de borde.
* Generación de snapshots del campo de onda y sismograma sintético en 52 receptores (`main_solver.py`).

#### 🚀 Fase 3: Mejoras y Conclusiones (`avance 3` / Etapa Final)
* Sustitución del *sponge layer* por bordes absorbentes **PML** (*Perfectly Matched Layer*).
* Incorporación de modelos de velocidad complejos (interfaces inclinadas, anomalías).
* Avance hacia el problema inverso (inversión tomográfica / *Full Waveform Inversion*).
* Redacción final del informe del proyecto utilizando el editor colaborativo Overleaf (LaTeX) para la estructuración de resultados y conclusiones.

---

## 🛠️ Herramientas y Tecnologías Utilizadas
* **Programación Científica:** Python (estructuración de arreglos y vectorización intensiva con `numpy`).
* **Documentación:** LaTeX (Overleaf) para la elaboración rigurosa de informes académicos y formato de notación matemática. 

---

**Autor:** Benjamín Ignacio Medina Sánchez
