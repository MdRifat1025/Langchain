from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from transformers import pipeline

load_dotenv()

# Model 1: Groq (for notes and merging)
model1 = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Model 2: Hugging Face (for quiz generation)
pipe = pipeline("text2text-generation", model="google/flan-t5-small")
hf_pipeline = HuggingFacePipeline(pipeline=pipe)
model2 = ChatHuggingFace(llm=hf_pipeline)

# Prompts
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question-answer pairs from the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document.\nNotes -> {notes}\nQuiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

# Parser
parser = StrOutputParser()

# Run notes + quiz generation in parallel
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

# Merge notes and quiz into a single output
merge_chain = prompt3 | model1 | parser

# Full pipeline
chain = parallel_chain | merge_chain

# Input text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Run chain
result = chain.invoke({"text": text})

print("\n=== Final Merged Output ===\n")
print(result)

# Print chain structure
print("\n=== Chain Graph ===\n")
chain.get_graph().print_ascii()
