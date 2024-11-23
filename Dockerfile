FROM python:3.7

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt
RUN apt-get update

ENV PORT 8501

CMD ["streamlit","run","app.py"]