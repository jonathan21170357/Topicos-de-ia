import ollama
from agenteLocal import AgenteLocal

def iniciar_debate():
    # 1. Definición de los roles y límites de los 5 agentes
    # Se añade una restricción estricta de longitud para evitar alucinaciones
    restriccion = "Responde siempre de manera concisa en un máximo de 2 a 3 oraciones cortas. No alucines, mantén la coherencia del debate."
    
    prompts = {
        "Moderador": f"Eres el Moderador de un panel. Guías la conversación, mantienes el orden y haces preguntas directas al resto. {restriccion}",
        "Científico": f"Eres un Científico. Analizas todo desde una perspectiva lógica, basada en datos, física y matemáticas. {restriccion}",
        "Filósofo": f"Eres un Filósofo. Cuestionas la ética, la moral y el significado profundo de las cosas. {restriccion}",
        "Comediante": f"Eres un Comediante. Aportas humor, sarcasmo o chistes ligeros sobre lo que dicen los demás. {restriccion}",
        "Pragmático": f"Eres un Pragmático. Te enfocas en cómo monetizar la idea, hacerla útil, rápida y rentable en el mundo real. {restriccion}"
    }

    # 2. Inicialización de los 5 agentes
    print("Inicializando agentes...")
    agentes = {}
    for nombre, prompt in prompts.items():
        # Usamos el modelo gemma3:1b como viene por defecto en tu clase
        agentes[nombre] = AgenteLocal(system_prompt=prompt)
    
    # Lista para mantener el orden de los turnos
    orden_turnos = list(prompts.keys())

    # 3. Solicitar el tema al usuario
    print("\n" + "="*50)
    tema = input("Ingresa el tema sobre el cual debatirán los agentes: ")
    print("="*50 + "\n")

    # 4. Bucle de conversación (25 turnos en total)
    historial_debate = []
    lineas_totales = 25
    
    # Mensaje inicial para arrancar la cadena
    ultimo_mensaje = f"Hola a todos, hoy estamos aquí para debatir sobre: {tema}. ¿Qué opinan al respecto?"
    nombre_actual = "Moderador"
    
    print(f"**{nombre_actual}**: {ultimo_mensaje}\n")
    historial_debate.append(f"{nombre_actual}: {ultimo_mensaje}")

    agente_anterior = "Moderador"

    for i in range(1, lineas_totales):
        # Determinar de quién es el turno
        siguiente_idx = i % 5
        nombre_actual = orden_turnos[siguiente_idx]

        # Construir el prompt para que el agente sepa qué le acaban de decir
        prompt_entrada = f"El participante '{agente_anterior}' acaba de decir: '{ultimo_mensaje}'. Responde a esto siguiendo estrictamente tu rol."

        # Generar respuesta
        respuesta = agentes[nombre_actual].chat(prompt_entrada)

        # Mostrar y guardar la respuesta
        print(f"**{nombre_actual}**: {respuesta}\n")
        historial_debate.append(f"{nombre_actual}: {respuesta}")

        # Actualizar variables para la siguiente iteración
        agente_anterior = nombre_actual
        ultimo_mensaje = respuesta

    # 5. Generar Resumen y Calificación
    print("="*50)
    print("Generando resumen y evaluación final (esto puede tomar unos segundos)...")
    
    transcripcion = "\n".join(historial_debate)
    
    prompt_evaluador = (
        "Eres un evaluador experto en IA. A continuación te presentaré la transcripción de un debate "
        "entre 5 inteligencias artificiales con diferentes personalidades. "
        "Tu tarea es:\n"
        "1. Escribir un resumen conciso de los puntos principales discutidos.\n"
        "2. Otorgar una calificación del 1 al 10 sobre cómo se comportaron los agentes "
        "(¿Mantuvieron su rol? ¿Evitaron alucinar? ¿Fue fluida la conversación?).\n\n"
        f"Transcripción del debate:\n{transcripcion}"
    )

    # Creamos un agente exclusivo para evaluar
    evaluador = AgenteLocal(system_prompt="Eres un analista objetivo, estructurado y claro.")
    resultado_evaluacion = evaluador.chat(prompt_evaluador)

    print("\n" + "="*20 + " EVALUACIÓN FINAL " + "="*20)
    print(resultado_evaluacion)
    print("="*58)

if __name__ == "__main__":
    iniciar_debate()