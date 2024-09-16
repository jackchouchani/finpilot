import os
import argparse
import os
import base64


def lire_fichier(chemin):
    try:
        with open(chemin, 'r', encoding='utf-8') as fichier:
            return fichier.read()
    except Exception as e:
        return f"Erreur lors de la lecture du fichier : {str(e)}"

def compiler_fichiers(dossier_racine, extensions, dossiers_exclus, fichier_sortie):
    with open(fichier_sortie, 'w', encoding='utf-8') as sortie:
        for dossier_actuel, sous_dossiers, fichiers in os.walk(dossier_racine):
            # Exclure les dossiers spécifiés
            sous_dossiers[:] = [d for d in sous_dossiers if d not in dossiers_exclus]
            
            for fichier in fichiers:
                if any(fichier.endswith(ext) for ext in extensions):
                    chemin_complet = os.path.join(dossier_actuel, fichier)
                    contenu = lire_fichier(chemin_complet)
                    
                    sortie.write(f"{chemin_complet} :\n")
                    sortie.write("<<<\n")
                    sortie.write(contenu)
                    sortie.write("\n>>>\n\n")

def main():
    parser = argparse.ArgumentParser(description="Compiler des fichiers de code dans un fichier texte.")
    parser.add_argument("dossier", help="Le dossier racine à parcourir")
    parser.add_argument("--extensions", nargs='+', default=['.py'], help="Les extensions de fichier à inclure")
    parser.add_argument("--exclure", nargs='+', default=['node_modules'], help="Les dossiers à exclure")
    parser.add_argument("--sortie", default="compilation_code.txt", help="Le nom du fichier de sortie")
    
    args = parser.parse_args()
    
    compiler_fichiers(args.dossier, args.extensions, args.exclure, args.sortie)
    print(f"Compilation terminée. Résultat dans {args.sortie}")

if __name__ == "__main__":
    main()