export OPENAI_API_KEY=<your_api_key>
export OPENAI_ORGANIZATION_IDS=<your_organization_id>  # Optional; if not set, this will be your default org id.

python evaluate.py --model_name='<model_name>' \
    --reference='text-davinci-003' \
    --evaluator='gpt-4-0613' \
