from regr.graph import Graph, Concept, Relation


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    graph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './')

    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        sentence = Concept(name='sentence')

        pair = Concept(name='pair')
        pair.has_a(word, word)

    with Graph('application') as app_graph:
        people = Concept(name='people')
        organization = Concept(name='organization')

        people.is_a(word)
        organization.is_a(word)

        #people.not_a(organization)
        #organization.not_a(people)

        work_for = Concept(name='work_for')
        work_for.is_a(pair)
        work_for.has_a(people, organization)

        kill = Concept(name='kill')
        kill.is_a(pair)
        kill.has_a(people, people)
