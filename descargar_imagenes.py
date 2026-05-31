import os
import time
import requests
from io import BytesIO
from PIL import Image

ruta_base = os.getcwd()
carpeta_principal = os.path.join(ruta_base, "dataset_png")
os.makedirs(carpeta_principal, exist_ok=True)

categorias = {
    "carros": "car",
    "casas": "house building",
    "motos": "motorcycle"
}

print("--- Construyendo Dataset (con Control de Velocidad) ---")
# Un User-Agent que simula un navegador real para evitar filtros estrictos
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
url_api = "https://commons.wikimedia.org/w/api.php"

for nombre, busqueda in categorias.items():
    print(f"\n📥 Descargando categoría: {nombre}...")
    ruta_guardado = os.path.join(carpeta_principal, nombre)
    os.makedirs(ruta_guardado, exist_ok=True)

    validas_guardadas = 0
    offset = 0

    while validas_guardadas < 100:
        parametros = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": busqueda, 
            "gsrnamespace": 6, 
            "gsrlimit": 50,
            "gsroffset": offset,
            "prop": "imageinfo",
            "iiprop": "url"
        }

        try:
            # Hacemos la petición a la API
            respuesta_api = requests.get(url_api, params=parametros, headers=headers)
            
            # Verificamos que el servidor nos haya respondido correctamente (Código 200)
            if respuesta_api.status_code != 200:
                print("⏳ El servidor nos pidió esperar. Pausando 5 segundos...")
                time.sleep(5)
                continue # Volvemos a intentar la misma petición

            # Ahora es seguro convertir a JSON
            respuesta = respuesta_api.json()
            paginas = respuesta.get("query", {}).get("pages", {})

            if not paginas:
                busqueda = busqueda + " photo"
                offset = 0
                continue

            for page_id, info in paginas.items():
                if validas_guardadas >= 100:
                    break
                
                image_info = info.get("imageinfo", [])
                if image_info:
                    img_url = image_info[0].get("url")
                    
                    if any(img_url.lower().endswith(ext) for ext in ['.svg', '.ogg', '.ogv', '.pdf', '.webm']):
                        continue

                    try:
                        respuesta_img = requests.get(img_url, headers=headers, timeout=5)
                        
                        if respuesta_img.status_code == 200:
                            img_bytes = BytesIO(respuesta_img.content)
                            img_final = Image.open(img_bytes)
                            
                            if img_final.mode != "RGB":
                                img_final = img_final.convert("RGB")
                            
                            ruta_archivo = os.path.join(ruta_guardado, f"{nombre}_{validas_guardadas+1}.png")
                            img_final.save(ruta_archivo, "PNG")
                            
                            print(f"  [+] Guardada: {nombre}_{validas_guardadas+1}.png")
                            validas_guardadas += 1
                            
                            # PAUSA DE CORTESÍA: Evita que nos bloqueen al descargar imágenes
                            time.sleep(0.2) 
                            
                    except Exception:
                        pass
                        
            offset += 50
            # PAUSA DE CORTESÍA: Evita que nos bloqueen la API de búsqueda
            time.sleep(1) 
            
        except requests.exceptions.JSONDecodeError:
            print("⏳ Error de decodificación (Posible bloqueo temporal). Pausando 5 segundos...")
            time.sleep(5)
        except Exception as e:
            print(f"❌ Error inesperado: {e}. Reintentando...")
            time.sleep(3)
            
    print(f"✅ Categoría '{nombre}': {validas_guardadas} imágenes PNG listas.")

print("\n✨ ¡Dataset completado! Tienes exactamente 100 imágenes PNG por categoría.")