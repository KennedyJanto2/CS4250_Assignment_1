#-------------------------------------------------------------------------
# AUTHOR: Kennedy Janto
# FILENAME: search_engine.py
# SPECIFICATION: description of the program
# FOR: CS 4250- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard arrays

#importing some Python libraries
import csv
import math

documents = []
labels = []

#reading the data in a csv file
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])
            labels.append(row[1].strip())

#Conduct stopword removal.
stopWords = {'I', 'and', 'She', 'They', 'her', 'their'}
filtered_documents = [' '.join([word for word in doc.split() if word.lower() not in stopWords]) for doc in documents]

#Conduct stemming.
steeming = {
  "cats": "cat",
  "dogs": "dog",
  "loves": "love",
}
stemmed_documents = [' '.join([steeming.get(word, word) for word in doc.split()]) for doc in filtered_documents]

#Identify the index terms.
terms = list(set(word for doc in stemmed_documents for word in doc.split()))

#Build the tf-idf term weights matrix.
N = len(stemmed_documents)
idf = {}
for term in terms:
    df = sum(1 for doc in stemmed_documents if term in doc.split())
    idf[term] = math.log(N / (df if df else 1))

docMatrix = []
for doc in stemmed_documents:
    tfidf_weights = [doc.split().count(term) * idf[term] for term in terms]
    docMatrix.append(tfidf_weights)

#Calculate the document scores (ranking) using document weigths (tf-idf) calculated before and query weights (binary - have or not the term).
query = {'cat', 'dog'}
query_weights = [1 if term in query else 0 for term in terms]
docScores = [sum(a*b for a, b in zip(doc_weights, query_weights)) for doc_weights in docMatrix]

#Calculate and print the precision and recall of the model by considering that the search engine will return all documents with scores >= 0.1.
retrieved_docs = [i for i, score in enumerate(docScores) if score >= 0.1]
relevant_docs = [i for i, label in enumerate(labels) if label == "R"]
relevant_retrieved_docs = set(retrieved_docs).intersection(relevant_docs)

precision = len(relevant_retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
recall = len(relevant_retrieved_docs) / len(relevant_docs) if relevant_docs else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")