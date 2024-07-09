FROM python:3.11
WORKDIR /app
COPY . .
# install ffmpeg

RUN apt-get update
RUN apt-get install -y ffmpeg
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN apt-get install -y libgl1-mesa-glx

# install python dependencies
RUN pip install --upgrade pip

# install spacy

RUN pip install spacy

# download spacy model
RUN python -m spacy download en_core_web_sm

# spanish model
RUN python -m spacy download es_core_news_sm



RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]