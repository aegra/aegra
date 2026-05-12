import { StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ChatState, type ChatStateType } from "./state.js";

const model = new ChatOpenAI({
  modelName: process.env.MODEL_NAME || "gpt-4o-mini",
  temperature: 0,
});

/**
 * Chatbot node — invokes the LLM with the current message history.
 */
async function chatbot(state: ChatStateType): Promise<Partial<ChatStateType>> {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
}

// Build the graph (uncompiled StateGraph)
const builder = new StateGraph(ChatState)
  .addNode("chatbot", chatbot)
  .addEdge("__start__", "chatbot")
  .addEdge("chatbot", "__end__");

/**
 * Uncompiled graph — exported for Aegra to load.
 *
 * The Aegra JS bridge compiles this with a native PostgresSaver
 * checkpointer, enabling interrupt, resume, and time-travel support.
 *
 * Configure in aegra.json:
 * {
 *   "graphs": {
 *     "js_chatbot": {
 *       "runtime": "langgraphjs",
 *       "path": "./examples/langgraphjs_chatbot/graph.ts:graph"
 *     }
 *   }
 * }
 */
export const graph = builder;
