import readline from 'readline';
import { Ollama } from '@langchain/community/llms/ollama';
import { PromptTemplate } from '@langchain/core/prompts';

import { BufferMemory } from 'langchain/memory';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const ask = (q) => new Promise(resolve => rl.question(q, resolve));

const vectorStore = await HNSWLib.load('./vectorstore', new OllamaEmbeddings({ model: 'nomic-embed-text' }));

const llm = new Ollama({ model: 'phi4-mini' });
const memory = new BufferMemory();

const template = `
You are a GraphQL expert helping the user write valid queries/mutations. and answer questions about it.

STRICT RULES:
- Only use fields and types explicitly found in the context.
- Do NOT guess or invent anything.
- If unsure, say "I need more information".

Conversation so far:
{history}

Relevant schema context:
{context}

User request:
{input}

Reply with a valid GraphQL query or mutation based on the schema and context above.
RESPONSE FORMAT:
- GraphQL Query/Mutation
- Type Definition
- Comments if needed
- NO guesses or assumptions
- example, with mandatory fields filled
`;

const prompt = new PromptTemplate({
  template,
  inputVariables: ['history', 'context', 'input'],
});

async function run() {
  while (true) {
    const userInput = await ask('You> ');
    if (userInput.toLowerCase() === 'exit') break;

    const relatedDocs = await vectorStore.similaritySearch(userInput, 20);
    const context = relatedDocs.map(d => d.pageContent).join('\n');
    if (context === '') {
        console.log("I couldn't find relevant schema context. Please refine your question.")
        return ;
      }

    const history = await memory.loadMemoryVariables({});
    const finalPrompt = await prompt.format({
      history: history.chat_history ?? '',
      context,
      input: userInput,
    });
    const response = await llm.invoke(finalPrompt);
    console.log("\Assistant>\n", response.trim());

    await memory.saveContext({ input: userInput }, { output: response });
  }

  rl.close();
}

run();
