# FORECASTING
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

### SETUP

```sh
$ echo seteo de tu ambiente de trabajo
git clone git@github.com:alaya-digital-solutions/challenges
$ cd challenges
$ echo usa tu ambiente predefinido, venv, conda u otro para instalar los requirements
$ pip install -r requirements.txt
```

### PROBLEMA Y SOLUCIÓN BUSCADA
El contexto: Los datos son una muestra sintetica de supply chain en el cual tienes la cantidad de ventas realizadas en el tiempo de dos articulos, el objetivo de este problema es que seas capas de predecir la demanda de estos articulos en las fechas que son señaladas como test. Para ello, se espera que puedas realizar modelos de series de tiempo sobre estos articulos. Este es un problema muy común en  muchas industrias (retail, manufactura, transporte, energía, etc), por lo que en tu día a día estarás enfrentado a este tipo de problemas.

### PREGUNTAS
Peguntas que nos interesan saber ! 

1. Análisis de ciclicidad de ambas series, es estacionaría? no es estacionaría?
2. ¿Cual debierá ser la frecuencia de predicción?
3. ¿Cual es el número apropiado de timestamps?, estimación visual
4. Es factible el deep learning en alguna de las series?
5. Escribe el código
6. Es escalable tu solución ¿Que dificultades tuvo el modelamiento? ¿Como harías escalable tu solución? 
7. ¿Tus resultados son reproducibles en el tiempo?

### QUE ENTREGAR?

Un .csv con la columna sku y las predicciones de las fechas que vienen señaladas como test en la data de entrenamiento
ver archivo de ejemplo

### DICCIONARIO DE DATOS
* sku: nombre del sku vendido
* cantidad_vendida: cantidad de unidades vendidas de este sku
* fecha: fecha en la que se vendió esa cantidad

### ANEXO
Este es un problema más que resuelto en muchas industrias, sobre todo en Retail. Este desafío propone encontrar nuevas maneras de abordar estas problemáticas, que son comunes a una diversa gamma de industrias.

