txt = """
        Alter: 62
        Rasse:
        - Mensch
        Kulturgruppe:
        - Mittelreich
        Beruf:
        - Alchemist
        - Wunderheiler
        Orte:
        - "[[Neu Havena]]"
        Fraktionen:
        - "[[Das Kaiserreich]]"
        Personen:
        - "[[Wenzel]]"
        - "[[Gregor Baldinger]]"
        - "[[Faktotum]]"
        - "[[Caruso 'der Schillernde']]"
        tags:
        Bild: 01 Data/Bilder/Portraits/Barnabas.png
        ---
        ![[Banner - Person.jpg|banner]]

        ---
        > [!infobox|right]
        > ```meta-bind
        > INPUT[Portrait][]
        > ```
        > |   |   |
        > | - | - |
        > | **Alter** | `VIEW[{Alter}][text]` |
        > | **Rasse** | `VIEW[{Rasse}][text]` |
        > | **Kultur** | `VIEW[{Kulturgruppe}][text]` |
        > | **Beruf** | `VIEW[{Beruf}][text]` |
        > ---
        > |   |   |
        > | - | - |
        > | **Angehörigkeit** | `VIEW[{Fraktionen}][link]` |
        > | **Anzutreffen** | `VIEW[{Orte}][link]` |
        > | **Beziehungen** | `VIEW[{Personen}][link]` |

        Allgemeine Informationen zur Person.

        ##### Motivation
        Einen festen Platz finden und akzeptiert werden.

        ##### Aussehen
        Kahle Stelle auf dem Kopf, stellenweise weiß graues haar, wirrer Bart, grauer Starr im linken Auge, leicht gebeugte Haltung, unterdurchschnittlich groß, braune Kutte

        ##### Charakter
        Intelligent, Trickbetrüger, Gaukler, Schnell angegriffen, gibt sich Schwächer als er ist
"""

from deep_translator import GoogleTranslator


print(GoogleTranslator(source="de", target="en").translate(txt))