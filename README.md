# Física Computacional - FIS 205

Este repositorio contiene los informes y códigos de proyectos y tareas desarrolladas durante el curso de Física Computacional.

---

## Proyecto Semestral: Propagación de Ondas Sísmicas 2D

**Descripción:** Simulación numérica de la propagación de ondas sísmicas acústicas en medios geológicos heterogéneos por capas, basada en el trabajo clásico de *Virieux (1986)*. El proyecto implementa un solver de la ecuación de onda 2D mediante diferencias finitas de **4º orden en espacio** y **2º orden en tiempo**, excitado por un pulso de **Ricker** (25 Hz), con condiciones de frontera absorbentes tipo *sponge layer*. El objetivo final es generar sismogramas sintéticos que sirvan de base para una futura inversión tomográfica.

### Estado de Avance (Segunda Entrega - 70%)

A continuación se detalla el progreso de los objetivos computacionales y físicos:

#### Fase 1: Modelo Físico y Fuente (Completado)

- [x] Formulación de la ecuación de onda acústica 2D en medio heterogéneo.
- [x] Implementación del pulso de Ricker con frecuencia central de 25 Hz (`fuente.py`).
- [x] Verificación del contenido espectral de banda limitada (FFT del pulso).
- [x] Corrección del retardo temporal `t0 = 1/fp` para garantizar causalidad y evitar truncamiento.

#### Fase 2: Esquema Numérico y Simulación (Completado)

- [x] Discretización del Laplaciano con estarcido (*stencil*) de 4º orden, vectorizado con `numpy`.
- [x] Integración temporal explícita tipo *leapfrog* de 2º orden.
- [x] Verificación automática del criterio de estabilidad **CFL** (Courant = 0.40).
- [x] Construcción y lectura del modelo de velocidad por capas en formato `.npy` (`modelo.py`).
- [x] Implementación de capa absorbente (*sponge layer*) para reducir reflexiones de borde.
- [x] Generación de snapshots del campo de onda y sismograma sintético en 52 receptores (`main_solver.py`).

#### Fase 3: Mejoras y Conclusiones (Pendiente - Próxima Entrega)

- [ ] Sustitución del *sponge layer* por bordes absorbentes **PML** (*Perfectly Matched Layer*).
- [ ] Incorporación de modelos de velocidad complejos (interfaces inclinadas, anomalías).
- [ ] Avance hacia el problema inverso (inversión tomográfica / *Full Waveform Inversion*).
- [ ] Redacción final del informe en LaTeX y conclusiones.



