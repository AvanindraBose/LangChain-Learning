from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#  Wbe based loader is best for Static Web Pages

loader = WebBaseLoader(web_path='https://www.amazon.in/JBL-Bluetooth-Dustproof-PartyBoost-Personalization/dp/B09HGSCL9Q/?_encoding=UTF8&pd_rd_w=RRBI8&content-id=amzn1.sym.340182bc-8d5c-49c7-8b69-c0403f7ba3a7%3Aamzn1.symc.752cde0b-d2ce-4cce-9121-769ea438869e&pf_rd_p=340182bc-8d5c-49c7-8b69-c0403f7ba3a7&pf_rd_r=59W98QPBVD9T0YBDR7WA&pd_rd_wg=saGgk&pd_rd_r=c3a1da95-7a86-47e5-9137-67e8105f5253&ref_=pd_hp_d_atf_ci_mcx_mr_&th=1')

docs = loader.load()

model = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the questions \n {question} for this text \n {text}",
    input_variables=["question","text"]
)

chain = prompt | model | parser

result = chain.invoke({'question':'tell me 2 good reviews and 2 bad reviews for this product','text': docs[0].page_content})

print(result)