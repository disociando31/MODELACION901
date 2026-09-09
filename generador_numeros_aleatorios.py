"""
==============================================================================
 GENERADOR DE NUMEROS PSEUDOALEATORIOS
 Actividad S4 - Modelacion - Ingenieria de Sistemas y Computacion
 Universidad de Cundinamarca I

 Este modulo implementa DOS algoritmos de generacion de numeros
 pseudoaleatorios:

   1. Generador Congruencial Lineal (LCG)      -> algoritmo CONGRUENCIAL
   2. Metodo de Cuadrados Medios (Middle-Square) -> algoritmo NO CONGRUENCIAL

 y valida la aleatoriedad de las secuencias producidas mediante:

   - Test de Chi-cuadrado (uniformidad de la distribucion)
   - Test de Kolmogorov-Smirnov (uniformidad de la distribucion, version
     no parametrica sobre la funcion de distribucion acumulada)
   - Test de autocorrelacion lag-1 (independencia entre valores consecutivos)
==============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import List

from scipy import stats


# ==============================================================================
# 1. ALGORITMO CONGRUENCIAL: GENERADOR CONGRUENCIAL LINEAL (LCG)
# ==============================================================================
#
# Formula matematica:
#       X(n+1) = (a * X(n) + c) mod m
#
# Donde:
#   X0  : semilla (valor inicial, 0 <= X0 < m)
#   a   : multiplicador
#   c   : incremento
#   m   : modulo (define el rango maximo de valores: 0 .. m-1)
#
# El numero pseudoaleatorio en el rango [0, 1) se obtiene como:
#       U(n) = X(n) / m
#
# Condiciones para que el generador tenga PERIODO MAXIMO (teorema de Hull-
# Dobell), es decir, que recorra los m valores posibles antes de repetirse:
#   1) mcd(c, m) = 1                    (c y m son primos relativos)
#   2) (a - 1) es divisible por todos los factores primos de m
#   3) (a - 1) es divisible por 4 si m es divisible por 4
#
# Usamos los parametros clasicos de Numerical Recipes (ANSI C):
#   a = 1664525, c = 1013904223, m = 2**32
# que cumplen el teorema de Hull-Dobell y son ampliamente usados en la
# practica por su buen comportamiento estadistico.
# ==============================================================================

class GeneradorCongruencialLineal:
    """Generador Congruencial Lineal (LCG) - algoritmo CONGRUENCIAL."""

    def __init__(self, semilla: int, a: int = 1664525, c: int = 1013904223,
                 m: int = 2 ** 32):
        if not (0 <= semilla < m):
            raise ValueError("La semilla debe cumplir 0 <= semilla < m")
        self.semilla_inicial = semilla
        self.a = a
        self.c = c
        self.m = m
        self.estado = semilla

    def siguiente(self) -> float:
        """Genera el siguiente numero pseudoaleatorio en [0, 1)."""
        self.estado = (self.a * self.estado + self.c) % self.m
        return self.estado / self.m

    def generar_secuencia(self, cantidad: int, minimo: float = 0.0,
                           maximo: float = 1.0) -> List[float]:
        """Genera una secuencia de 'cantidad' numeros en el rango [minimo, maximo)."""
        secuencia = []
        for _ in range(cantidad):
            u = self.siguiente()
            valor = minimo + u * (maximo - minimo)
            secuencia.append(valor)
        return secuencia

    def reiniciar(self):
        """Reinicia el generador a su semilla original (reproducibilidad)."""
        self.estado = self.semilla_inicial


# ==============================================================================
# 2. ALGORITMO NO CONGRUENCIAL: METODO DE CUADRADOS MEDIOS (MIDDLE-SQUARE)
# ==============================================================================
#
# Propuesto por John von Neumann (1946). Es NO congruencial porque no usa
# una relacion de recurrencia lineal con modulo; en su lugar, eleva al
# cuadrado el numero anterior y extrae los digitos centrales.
#
# Algoritmo (para numeros de n digitos, tipicamente n par):
#   1) Se parte de una semilla X0 de n digitos.
#   2) Se eleva al cuadrado: Y = X(n)^2   -> resultado de hasta 2n digitos
#      (se rellena con ceros a la izquierda si es necesario).
#   3) Se extraen los n digitos centrales de Y; ese es el nuevo X(n+1).
#   4) El numero pseudoaleatorio en [0, 1) es X(n+1) / 10^n.
#
# Limitaciones conocidas (importantes para la discusion en el informe):
#   - Tiene periodos cortos y puede degenerar en 0 o entrar en ciclos
#     pequeños dependiendo de la semilla.
#   - Es sensible a la eleccion de la semilla: algunas semillas producen
#     secuencias de muy baja calidad estadistica.
#   - Por eso hoy en dia es un metodo principalmente didactico/historico,
#     util para ilustrar el concepto de "no congruencial" y sus problemas.
# ==============================================================================

class GeneradorCuadradosMedios:
    """Metodo de Cuadrados Medios (Middle-Square) - algoritmo NO CONGRUENCIAL."""

    def __init__(self, semilla: int, n_digitos: int = 4):
        self.n_digitos = n_digitos
        self.limite = 10 ** n_digitos
        if not (0 < semilla < self.limite):
            raise ValueError(f"La semilla debe tener {n_digitos} digitos "
                              f"(0 < semilla < {self.limite})")
        self.semilla_inicial = semilla
        self.estado = semilla
        self._contador_degeneracion = 0

    def siguiente(self) -> float:
        """Genera el siguiente numero pseudoaleatorio en [0, 1)."""
        cuadrado = self.estado ** 2
        # Se rellena con ceros a la izquierda hasta 2*n_digitos
        cuadrado_str = str(cuadrado).zfill(2 * self.n_digitos)
        inicio = self.n_digitos // 2
        centro = cuadrado_str[inicio: inicio + self.n_digitos]
        nuevo_estado = int(centro)

        # Salvaguarda didactica: si el generador degenera en 0 (limitacion
        # conocida del metodo), se reinyecta una perturbacion impar basada
        # en la semilla original para poder continuar la demostracion.
        if nuevo_estado == 0:
            self._contador_degeneracion += 1
            nuevo_estado = (self.semilla_inicial * (self._contador_degeneracion * 2 + 1)) % self.limite
            if nuevo_estado == 0:
                nuevo_estado = 1

        self.estado = nuevo_estado
        return self.estado / self.limite

    def generar_secuencia(self, cantidad: int, minimo: float = 0.0,
                           maximo: float = 1.0) -> List[float]:
        secuencia = []
        for _ in range(cantidad):
            u = self.siguiente()
            valor = minimo + u * (maximo - minimo)
            secuencia.append(valor)
        return secuencia

    def reiniciar(self):
        self.estado = self.semilla_inicial
        self._contador_degeneracion = 0


# ==============================================================================
# 3. PRUEBAS ESTADISTICAS DE VALIDACION DE ALEATORIEDAD
# ==============================================================================

@dataclass
class ResultadoPruebas:
    nombre_generador: str
    n: int
    chi2_estadistico: float
    chi2_p_valor: float
    chi2_conclusion: str
    ks_estadistico: float
    ks_p_valor: float
    ks_conclusion: str
    autocorrelacion_lag1: float
    autocorr_conclusion: str
    media: float
    varianza: float


def test_chi_cuadrado(datos: List[float], minimo: float, maximo: float,
                       k_intervalos: int = 10, alpha: float = 0.05):
    """
    Test de Chi-cuadrado de bondad de ajuste para UNIFORMIDAD.

    Se divide el rango [minimo, maximo] en k_intervalos de igual ancho.
    Se compara la frecuencia OBSERVADA en cada intervalo contra la
    frecuencia ESPERADA bajo una distribucion uniforme (n / k_intervalos).

            Estadistico: Chi2 = sum( (Oi - Ei)^2 / Ei )

    H0: los datos provienen de una distribucion uniforme.
    Si p_valor > alpha  -> NO se rechaza H0 (los datos son compatibles con
    una distribucion uniforme).
    """
    n = len(datos)
    ancho = (maximo - minimo) / k_intervalos
    frecuencias_obs = [0] * k_intervalos

    for x in datos:
        idx = int((x - minimo) / ancho)
        if idx == k_intervalos:  # el valor maximo cae justo en el borde
            idx -= 1
        frecuencias_obs[idx] += 1

    frecuencia_esperada = n / k_intervalos
    frecuencias_esp = [frecuencia_esperada] * k_intervalos

    chi2_stat, p_valor = stats.chisquare(f_obs=frecuencias_obs, f_exp=frecuencias_esp)
    conclusion = ("No se rechaza H0: la secuencia es compatible con una "
                  "distribucion uniforme.") if p_valor > alpha else \
                 ("Se rechaza H0: hay evidencia de que la secuencia NO es "
                  "uniforme.")
    return chi2_stat, p_valor, conclusion, frecuencias_obs


def test_kolmogorov_smirnov(datos: List[float], minimo: float, maximo: float,
                             alpha: float = 0.05):
    """
    Test de Kolmogorov-Smirnov de bondad de ajuste para UNIFORMIDAD.

    Compara la funcion de distribucion empirica de los datos contra la
    funcion de distribucion acumulada teorica de una Uniforme(minimo, maximo).

            Estadistico: D = max | Fn(x) - F(x) |

    H0: los datos provienen de una distribucion Uniforme(minimo, maximo).
    Si p_valor > alpha  -> NO se rechaza H0.
    """
    ks_stat, p_valor = stats.kstest(datos, 'uniform', args=(minimo, maximo - minimo))
    conclusion = ("No se rechaza H0: la secuencia es compatible con una "
                  "distribucion uniforme.") if p_valor > alpha else \
                 ("Se rechaza H0: hay evidencia de que la secuencia NO es "
                  "uniforme.")
    return ks_stat, p_valor, conclusion


def test_autocorrelacion_lag1(datos: List[float], alpha: float = 0.05):
    """
    Prueba de INDEPENDENCIA basada en el coeficiente de autocorrelacion
    de rezago 1 (lag-1), es decir, la correlacion de Pearson entre la
    secuencia X(n) y la secuencia desplazada X(n+1).

    Si los numeros son independientes, la autocorrelacion deberia ser
    cercana a 0. Se usa como referencia practica |r| < 0.1 (para muestras
    grandes) como umbral de "sin evidencia fuerte de dependencia".
    """
    x = datos[:-1]
    y = datos[1:]
    r, _ = stats.pearsonr(x, y)
    conclusion = ("No se observa evidencia fuerte de dependencia entre "
                  "valores consecutivos (|r| < 0.1).") if abs(r) < 0.1 else \
                 ("Se observa una posible dependencia entre valores "
                  "consecutivos (|r| >= 0.1); revisar el generador.")
    return r, conclusion


def validar_generador(nombre: str, datos: List[float], minimo: float,
                       maximo: float) -> ResultadoPruebas:
    chi2_stat, chi2_p, chi2_concl, _ = test_chi_cuadrado(datos, minimo, maximo)
    ks_stat, ks_p, ks_concl = test_kolmogorov_smirnov(datos, minimo, maximo)
    autocorr, autocorr_concl = test_autocorrelacion_lag1(datos)

    return ResultadoPruebas(
        nombre_generador=nombre,
        n=len(datos),
        chi2_estadistico=chi2_stat,
        chi2_p_valor=chi2_p,
        chi2_conclusion=chi2_concl,
        ks_estadistico=ks_stat,
        ks_p_valor=ks_p,
        ks_conclusion=ks_concl,
        autocorrelacion_lag1=autocorr,
        autocorr_conclusion=autocorr_concl,
        media=sum(datos) / len(datos),
        varianza=sum((x - sum(datos) / len(datos)) ** 2 for x in datos) / len(datos),
    )


def imprimir_resultado(r: ResultadoPruebas):
    print(f"\n{'=' * 70}")
    print(f" GENERADOR: {r.nombre_generador}   (n = {r.n})")
    print(f"{'=' * 70}")
    print(f" Media muestral       : {r.media:.5f}   (teorica esperada ~ 0.5)")
    print(f" Varianza muestral    : {r.varianza:.5f}   (teorica esperada ~ 0.0833)")
    print(f"\n --- Test de Chi-cuadrado (uniformidad) ---")
    print(f" Estadistico Chi2     : {r.chi2_estadistico:.5f}")
    print(f" p-valor              : {r.chi2_p_valor:.5f}")
    print(f" Conclusion           : {r.chi2_conclusion}")
    print(f"\n --- Test de Kolmogorov-Smirnov (uniformidad) ---")
    print(f" Estadistico D        : {r.ks_estadistico:.5f}")
    print(f" p-valor              : {r.ks_p_valor:.5f}")
    print(f" Conclusion           : {r.ks_conclusion}")
    print(f"\n --- Test de autocorrelacion lag-1 (independencia) ---")
    print(f" Coeficiente r        : {r.autocorrelacion_lag1:.5f}")
    print(f" Conclusion           : {r.autocorr_conclusion}")


# ==============================================================================
# 4. PROGRAMA PRINCIPAL (INTERFAZ DE CONSOLA)
# ==============================================================================

def ejecutar_demo(semilla: int = 12345, cantidad: int = 1000):
    """Ejecuta ambos generadores con los parametros dados y valida los resultados."""
    print("\n" + "#" * 70)
    print("# DEMOSTRACION: GENERACION Y VALIDACION DE NUMEROS PSEUDOALEATORIOS")
    print("#" * 70)
    print(f"\nParametros de ejecucion:")
    print(f"  Semilla   : {semilla}")
    print(f"  Cantidad  : {cantidad}")
    print(f"  Rango     : [0, 1)")

    # --- Generador Congruencial Lineal ---
    lcg = GeneradorCongruencialLineal(semilla=semilla)
    secuencia_lcg = lcg.generar_secuencia(cantidad, 0.0, 1.0)
    resultado_lcg = validar_generador("Congruencial Lineal (LCG)", secuencia_lcg, 0.0, 1.0)
    imprimir_resultado(resultado_lcg)

    # --- Generador de Cuadrados Medios ---
    # La semilla debe tener n_digitos digitos; se deriva de la semilla general.
    semilla_ms = int(str(semilla).zfill(4)[-4:])
    if semilla_ms % 2 == 0:
        semilla_ms += 1  # evitar semillas triviales
    ms = GeneradorCuadradosMedios(semilla=semilla_ms, n_digitos=4)
    secuencia_ms = ms.generar_secuencia(cantidad, 0.0, 1.0)
    resultado_ms = validar_generador("Cuadrados Medios (Middle-Square)", secuencia_ms, 0.0, 1.0)
    imprimir_resultado(resultado_ms)

    print("\n" + "#" * 70)
    print("# FIN DE LA DEMOSTRACION")
    print("#" * 70 + "\n")

    return secuencia_lcg, secuencia_ms, resultado_lcg, resultado_ms


def menu_interactivo():
    """Interfaz de consola simple para configurar y ejecutar los generadores."""
    print("\n--- GENERADOR DE NUMEROS PSEUDOALEATORIOS ---")
    print("Algoritmos disponibles:")
    print("  1) Congruencial Lineal (LCG)")
    print("  2) Cuadrados Medios (Middle-Square)")
    print("  3) Ambos (comparativo)")
    opcion = input("Seleccione una opcion [1/2/3]: ").strip() or "3"

    semilla = int(input("Ingrese la semilla (entero): ").strip() or "12345")
    cantidad = int(input("Cantidad de numeros a generar: ").strip() or "1000")
    minimo = float(input("Valor minimo del rango (Enter = 0.0): ").strip() or "0.0")
    maximo = float(input("Valor maximo del rango (Enter = 1.0): ").strip() or "1.0")

    if opcion in ("1", "3"):
        lcg = GeneradorCongruencialLineal(semilla=semilla)
        datos = lcg.generar_secuencia(cantidad, minimo, maximo)
        resultado = validar_generador("Congruencial Lineal (LCG)", datos, minimo, maximo)
        imprimir_resultado(resultado)

    if opcion in ("2", "3"):
        semilla_ms = int(str(semilla).zfill(4)[-4:]) or 1234
        if semilla_ms % 2 == 0:
            semilla_ms += 1
        ms = GeneradorCuadradosMedios(semilla=semilla_ms, n_digitos=4)
        datos = ms.generar_secuencia(cantidad, minimo, maximo)
        resultado = validar_generador("Cuadrados Medios (Middle-Square)", datos, minimo, maximo)
        imprimir_resultado(resultado)


if __name__ == "__main__":
    # Ejecuta automaticamente la demostracion con valores por defecto.
    # Para usar la interfaz interactiva de consola, comenta la linea de
    # abajo y descomenta menu_interactivo().
    ejecutar_demo(semilla=12345, cantidad=1000)
    # menu_interactivo()
