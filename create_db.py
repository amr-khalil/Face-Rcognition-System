import os, sys, sqlite3

# Existenz feststellen
#if os.path.exists("face.db"):
    #print("Datei bereits vorhanden")
    #sys.exit(0)

# Verbindung zur Datenbank erzeugen
connection = sqlite3.connect("face.db")

# Datensatz-Cursor erzeugen
cursor = connection.cursor()

def create_table():
  # Datenbanktabelle erzeugen
  sql = "CREATE TABLE IF NOT EXISTS people(person_id INTEGER PRIMARY KEY, name TEXT, mask TEXT, mydate TEXT, mytime TEXT)" 
  cursor.execute(sql)


create_table()
# Verbindung beenden
cursor.close()
connection.close()


