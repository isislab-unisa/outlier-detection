from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery("""
 PREFIX wdt: <http://www.wikidata.org/prop/direct/>
select distinct * where {
?s dbo:deathDate ?d.
?s owl:sameAs ?s_2.
filter(regex(str(?s_2), "^http://www.wikidata.org/"))
}
""")

sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    print(result["s"]["value"],result["d"]["value"],result["s_2"]["value"])


