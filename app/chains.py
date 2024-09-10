import os

from IPython.core.debugger import prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
    model_name ="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv('groq_api_key'),

    )

    def extract_jobs(self,cleaned_text) :
        prompt_extract = PromptTemplate.from_template('''###scraped text from website :
        {page_data}
        ###instruction:
        the scraped text is from the careers page of a website.
        your job is to extract the job postings and return them in JSON format container.
        following keys :'role' , 'experience' , 'skills' and 'description'.
        ###only return valid JSON (NO PREAMBLE):
        '''
                                                      )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data': cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException('context too big. Unable to parse jobs.')
        return res if isinstance(res,list) else [res]

    def write_mail(self,job,links):
        prompt_email = PromptTemplate.from_template(
            '''
            ###JOB DESCRIPTION:
            {job_description}
    
            ### INSTRUCTION:
            You are Mohan, a business development executive at MAK INFO TECH.MAK INFO TECH is an AI & Software company dedicated the 
            seamless integration of business processes through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalabilty,
            process optimization,cost reduction, and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of MAK INFO TECH.
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase MAK INFO TECH's portfolio: {link_list}
            Remember you are Mohan, BDE at MAK INFO TECH.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
    
            '''
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({'job_description': str(job), 'link_list': links})
        return res.content


if __name__ == "__main__":
    print(os.getenv('groq_api_key'))