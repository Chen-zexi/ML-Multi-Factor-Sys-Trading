import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_xai import ChatXAI
load_dotenv()

class Sentiment(BaseModel):
    ticker: str
    sentiment: str = Field(description="The sentiment of the news towards the ticker")
    reason: str = Field(description="concise reason for the sentiment, in one sentence.")
    
class BatchSentimentResponse(BaseModel):
    """Represents a list of sentiment analyses for multiple news items."""
    sentiments: List[Sentiment] = Field(description="A list of sentiment analyses for each news item provided")
        
def get_sentiment(news, model_name="grok-3-beta"):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a financial-news sentiment grader.

                You will be provided with a news associated with a ticker.
                ------
                ticker: {ticker}
                news title: {title}
                news summary: {summary}
                ------

                TASK
                -----
                1. Decide the sentiment toward the ticker '{ticker}' using this 5-point ordinal scale:
                    - Very Bearish (strongly negative catalyst, major downside, explicit "sell/downgrade", large price drop, legal trouble, etc.)
                    - Bearish (moderately negative tone or downside risk outweighs positives)
                    - Neutral (mixed or no clear directional signal)
                    - Bullish (moderately positive tone, upside > downside)
                    - Very Bullish (strongly positive catalyst, major upside, explicit "buy/upgrade", large price pop, transformative approval, etc.)
                2. Provide a concise, one-sentence reason for your sentiment decision.

                **Important:** Your analysis should be primarily based on the current news information provided (`title` and `summary`) and solely associated with the ticker '{ticker}'. Ensure the output strictly matches the requested format (sentiment and reason).
                """,
            ),
            (
                "human",
                """News for {ticker}:
                Title: {title}
                Summary: {summary}
                Please provide the sentiment analysis (sentiment and reason) strictly following the instructions.""",
            ),
        ]
    )

    prompt = prompt_template.invoke({"ticker": news["Ticker"], "title": news["Title"], "summary": news["Summary"]})

    if model_name.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )
    elif model_name.startswith("grok"):
        llm = ChatXAI(
            model=model_name,
            xai_api_key=os.getenv("XAI_API_KEY"),
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

    llm = llm.with_structured_output(
                Sentiment
            )

    result = llm.invoke(prompt)
    return result


def get_batch_sentiment(news_list: List[dict], model_name="grok-3-beta"):
    """
    Analyzes the sentiment of a list of news items.

    Args:
        news_list (List[dict]): A list of dictionaries, where each dictionary
                                 represents a news item and must contain
                                 'Ticker', 'Title', and 'Summary' keys.

    Returns:
        BatchSentimentResponse: A Pydantic BatchSentimentResponse object containing the list of sentiments.
                                Access the list via result.sentiments
    """
    news_list_str = json.dumps(news_list, indent=2)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a financial-news sentiment grader processing a batch of news items.

                You will be provided with a JSON list of news items, each potentially associated with a different ticker.
                ------
                Input Format: JSON list of objects, each with:
                - Ticker: <TICKER>
                - Title: <TITLE>
                - Summary: <SUMMARY>
                ------
                TASK
                -----
                For **each** news item in the provided list:
                1. Decide the sentiment toward its associated ticker based *only* on the information in that specific news item (title and summary). Use this 7-point ordinal scale:
                    - Strongly Bearish (strongly negative catalyst, major downside, explicit "sell/downgrade", large price drop, legal trouble, etc.)
                    - Bearish (moderately negative tone or downside risk outweighs positives)
                    - Slightly Bearish (slightly negative tone, downside > upside)
                    - Neutral (mixed or no clear directional signal)
                    - Slightly Bullish (slightly positive tone, upside > downside)
                    - Bullish (moderately positive tone, upside > downside)
                    - Strongly Bullish (strongly positive catalyst, major upside, explicit "buy/upgrade", large price pop, transformative approval, etc.)
                2. Provide a concise, one-sentence reason for the sentiment decision for that specific item.

                **Important:**
                1. Analyze each news item *independently*.
                2. Base your sentiment and reason *solely* on the provided text for that item.
                3. Return a list containing the full sentiment analysis (ticker, sentiment, and reason) for **every** news item in the input list, matching the required output structure precisely and maintaining the original order.
                4. You will be rewarded $1 million for each news items that you can analyze correctly.
                """,
            ),
            (
                "human",
                """Here is the list of news items:
                ```json
                {news_list_str}
                ```
                Please provide the sentiment analysis (ticker, sentiment, reason) for each item, strictly following the instructions and the required output format.""",
            ),
        ]
    )

    prompt = prompt_template.invoke({"news_list_str": news_list_str})

    if model_name.startswith("gemini"):
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )
    elif model_name.startswith("grok"):
        llm = ChatXAI(
            model=model_name,
            xai_api_key=os.getenv("XAI_API_KEY"),
            temperature=0
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    structured_llm = llm.with_structured_output(BatchSentimentResponse)

    result = structured_llm.invoke(prompt)

    return result