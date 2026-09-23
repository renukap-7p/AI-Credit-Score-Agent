

import os
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types

# ----------------------------------------------------------------------
# LLM CONFIG — the customer/user never sees any of this. It just powers
# the "AI Agent" behind the scenes. Key comes from Google AI Studio.
#
# 1. Get a free key at https://aistudio.google.com/apikey
# 2. Set it as an environment variable BEFORE running this cell:
#   - In Google Colab: use the "Secrets" tab (key icon on the left) to
#     add a secret named GEMINI_API_KEY, then:
#         from google.colab import userdata
#         os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
#   - Locally: export GEMINI_API_KEY="AIza..." in your terminal
#
# Never hardcode the key directly in the script.
# ----------------------------------------------------------------------
LLM_MODEL = "gemini-flash-latest"
client = None
ENV_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if ENV_API_KEY:
    client = genai.Client(api_key=ENV_API_KEY)


def check_llm_connection():
    """Actually tests the configured Gemini key with a small API request."""
    if client is None:
        return ("🔴 **LLM Status: Offline** — no Gemini API key is configured. "
                "Paste a Google AI Studio key above and click Connect.")

    try:
        client.models.generate_content(
            model=LLM_MODEL,
            contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=5),
        )
        return (
            f"🟢 **LLM Status: Connected** — `{LLM_MODEL}` is reachable "
            "through the Gemini API."
        )
    except Exception as e:
        print(f"[LLM connection check failed] {e}")
        return (
            "🟡 **LLM Status: Key found, but Gemini rejected the connection.** "
            "Check that the key is valid, the Gemini API is enabled for that key, "
            "and the selected model is available. Then click Connect again."
        )


def connect_llm(api_key):
    """Called when the user pastes a key into the web UI and clicks
    Connect. Creates a fresh client with THAT key, does a real test
    call, and only wires it in globally if the call actually succeeds —
    so the badge can never lie about being connected."""
    global client

    if not api_key or not api_key.strip():
        client = None
        return "🔴 **LLM Status: Offline** — the API key field is empty. Paste your key and click Connect."

    try:
        candidate = genai.Client(api_key=api_key.strip())
        candidate.models.generate_content(
            model=LLM_MODEL,
            contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=5),
        )
        client = candidate  # only adopt it once the test call succeeds
        return f"🟢 **LLM Status: Connected** — live to `{LLM_MODEL}` via the Google AI Studio (Gemini) API."
    except Exception as e:
        client = None
        print(f"[LLM connect failed] {e}")  # full error stays in console/logs
        return (
            "🟡 **LLM Status: Connection failed.** "
            "The key was received, but Gemini rejected the test request. "
            "Check the API key, Gemini API access, and model availability."
        )


class EnhancedCreditAgent:
    """Enhanced AI Agent for credit optimization with visual plotting,
    structured analysis, and an LLM (Google Gemini, via Google AI Studio)
    layer that generates the natural-language narrative and answers
    follow-up questions."""

    # ------------------------------------------------------------------
    # Core deterministic scoring logic — unchanged. This is the "engine"
    # the LLM is told to reason over, so its advice stays grounded in
    # real numbers instead of hallucinating.
    # ------------------------------------------------------------------
    def analyze_profile(self, score, payment_history, utilization, history_length, new_credit):
        agent_logs = []
        recommendations = []

        # 1. Evaluate Core Status
        if score >= 750:
            status = "Excellent"
            rec_main = "Prime score status. Focus on maintaining low utilization to preserve high-tier interest rates."
        elif score >= 700:
            status = "Good"
            rec_main = "Solid financial profile. Targeted reductions in card balances will unlock prime tiers."
        elif score >= 650:
            status = "Fair"
            rec_main = "Average rating. Prioritize payment discipline and balance reductions to recover points."
        else:
            status = "Poor"
            rec_main = "Critical status. Immediate action is needed to resolve delinquent factors and manage balances."

        agent_logs.append(f"LOG: Classified base score {score} into '{status}' tier.")

        # 2. Rule Engine & Specific Interventions
        if utilization > 30:
            recommendations.append(f"⚠️ High Utilization ({utilization}%): Lower total card balances below 30% to avoid score penalties.")
            agent_logs.append(f"LOG: High utilization penalty detected ({utilization}% > 30%).")
        else:
            recommendations.append(f"✅ Healthy Utilization ({utilization}%): Maintaining balances below 30% reinforces your credit base.")
            agent_logs.append(f"LOG: Utilization check passed ({utilization}%).")

        if payment_history < 95:
            recommendations.append(f"⚠️ Payment History ({payment_history}%): Enable auto-pay. Recent late payments heavily weigh down your profile.")
            agent_logs.append(f"LOG: Payment delinquency flag triggered ({payment_history}% < 95%).")
        else:
            recommendations.append(f"✅ Strong Payment History ({payment_history}%): Consistent on-time payments are strengthening your score.")
            agent_logs.append(f"LOG: Payment record verified as healthy ({payment_history}%).")

        if history_length < 3:
            recommendations.append(f"⚠️ Short Credit Age ({history_length} yrs): Keep older lines open to extend your average credit age over time.")
            agent_logs.append(f"LOG: Credit age flagged as maturing ({history_length} yrs < 3 yrs).")

        if new_credit > 2:
            recommendations.append(f"⚠️ Frequent Inquiries ({new_credit} in 6 mos): Pause new credit applications to clear hard inquiries.")
            agent_logs.append(f"LOG: Inquiry velocity warning triggered ({new_credit} > 2).")

        # 3. Dynamic Point Projection Logic
        projected_gain = 0
        if utilization > 30: projected_gain += 25
        if payment_history < 100: projected_gain += 35
        if new_credit > 2: projected_gain += 15

        target_score = min(850, score + projected_gain)
        agent_logs.append(f"LOG: Calculated maximum potential recovery of +{projected_gain} pts. Target: {target_score}.")

        # 4. Generate Pandas Structural Table
        df = pd.DataFrame({
            "Credit Parameter": ["Credit Score", "Payment Record", "Utilization Rate", "Credit History Age", "Hard Inquiries"],
            "Current Profile": [f"{score}", f"{payment_history}%", f"{utilization}%", f"{history_length} Years", f"{new_credit}"],
            "Optimal Benchmark": ["750+", "95%+", "< 30%", "3+ Years", "< 2"],
            "Parameter Status": [
                status,
                "Optimal" if payment_history >= 95 else "Needs Attention",
                "Optimal" if utilization <= 30 else "Needs Attention",
                "Optimal" if history_length >= 3 else "Building Age",
                "Optimal" if new_credit <= 2 else "Too High"
            ]
        })

        # 5. Render Matplotlib Forecast Plot
        fig, ax = plt.subplots(figsize=(6, 3))
        categories = ['Current', 'Potential Gain', 'Target Score']
        values = [score, projected_gain, target_score]
        bar_colors = ['#2b5c8f', '#27ae60', '#8e44ad']

        bars = ax.bar(categories, values, color=bar_colors, width=0.45)
        ax.set_ylim(0, 950)
        ax.set_ylabel("Credit Score Points")
        ax.set_title("Forecasted Score Trajectory", fontsize=11, fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Save the latest state so the chatbot tab can answer questions
        # about it, and so the LLM call below has grounded numbers.
        self.last_state = {
            "score": score, "status": status, "payment_history": payment_history,
            "utilization": utilization, "history_length": history_length,
            "new_credit": new_credit, "projected_gain": projected_gain,
            "target_score": target_score, "recommendations": recommendations,
        }

        # 6. Build the narrative report — via the LLM when a key is
        # configured, falling back to the template if not. This call
        # also tells us the REAL outcome of this specific attempt
        # (not just whether a key exists), which drives the status badge.
        narrative, used_llm, error_msg = self._llm_generate_report(self.last_state, rec_main)

        report = f"### Overall Assessment: **{status}** ({score} Points)\n"
        report += narrative + "\n\n"
        report += "### Actionable Steps:\n" + "\n".join([f"- {r}" for r in recommendations]) + "\n\n"
        report += "### Agent Execution Logs:\n" + "\n".join([f"`{log}`" for log in agent_logs])
        report += f"\n\n`LOG: Narrative generation via {'Gemini (' + LLM_MODEL + ')' if used_llm else 'rule-based fallback'}.`"
        if error_msg:
            print(f"[LLM narrative generation failed] {error_msg}")  # detail stays in console only

        if used_llm:
            badge = f"🟢 **LLM Status: Connected** — this report was generated live by `{LLM_MODEL}`."
        elif client is None:
            badge = ("🔴 **LLM Status: Offline** — no Gemini API key is configured. "
                      "Agent is running on rule-based fallback only.")
        else:
            badge = (f"🟡 **LLM Status: Key found, but the call failed** "
                      f"({(error_msg or '')[:100]}). Used rule-based fallback for this report.")

        return report, df, fig, badge

    # ------------------------------------------------------------------
    # LLM call #1: turns the computed metrics into a grounded, personal
    # narrative paragraph. The model only sees numbers we calculated —
    # it is not allowed to invent scores, so the report stays accurate.
    # ------------------------------------------------------------------
    def _llm_generate_report(self, state, rec_main):
        fallback_text = (f"**Core Advice:** {rec_main}\n\n"
                          f"**Projected Score Growth:** +{state['projected_gain']} Points "
                          f"(Target Score: **{state['target_score']}**)")

        if client is None:
            return fallback_text, False, None

        prompt = f"""You are a credit-improvement AI agent. Using ONLY the data below
(never invent numbers), write a short, encouraging 3-4 sentence narrative
explaining the person's credit standing and what matters most right now.
End with one bolded line: a projected score range.

Score: {state['score']} ({state['status']})
Payment history: {state['payment_history']}%
Utilization: {state['utilization']}%
Credit history length: {state['history_length']} years
Hard inquiries (6 mo): {state['new_credit']}
Projected point gain if issues fixed: +{state['projected_gain']}
Target score: {state['target_score']}
"""
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=350),
            )
            return response.text.strip(), True, None
        except Exception as e:
            # Never let an API hiccup break the demo — fall back quietly,
            # but report the real error so the status badge is honest.
            return fallback_text, False, str(e)

    # ------------------------------------------------------------------
    # LLM call #2: the chatbot. Grounds every answer in the last
    # computed profile so it can't hallucinate numbers, and falls back
    # to the original keyword-matching logic if no key is configured
    # or the API call fails — so the demo never breaks.
    # ------------------------------------------------------------------
    def chat_reply(self, message, history):
        state = getattr(self, "last_state", None)
        if state is None:
            return "Please run **Run Agent Assessment** on the first tab first, then come back and ask me anything about your result."

        if client is not None:
            try:
                system_prompt = f"""You are a friendly, precise credit-improvement AI agent.
Answer the user's question using ONLY this profile data — never invent numbers:

Score: {state['score']} ({state['status']})
Payment history: {state['payment_history']}%
Utilization: {state['utilization']}%
Credit history length: {state['history_length']} years
Hard inquiries (6 mo): {state['new_credit']}
Projected point gain: +{state['projected_gain']}
Target score: {state['target_score']}
Recommendations: {"; ".join(state['recommendations'])}

Keep answers concise (2-4 sentences), warm, and actionable."""
                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=300,
                    ),
                )
                return response.text.strip()
            except Exception:
                pass  # fall through to rule-based backup below — chat never breaks

        # ---- rule-based fallback (also doubles as offline demo mode) ----
        msg = message.lower()

        if any(w in msg for w in ["util", "balance", "card"]):
            return (
                f"Your credit utilization is **{state['utilization']}%**. "
                + ("That's above the recommended 30% ceiling — paying down balances is usually "
                   "the fastest way to recover points." if state['utilization'] > 30
                   else "That's within the healthy range (under 30%) — keep it there.")
            )

        if any(w in msg for w in ["payment", "late", "pay"]):
            return (
                f"Your on-time payment record is **{state['payment_history']}%**. "
                + ("Even one more missed payment can undo months of progress — turning on "
                   "autopay is the highest-leverage fix here." if state['payment_history'] < 95
                   else "This is already strong — consistent on-time payments are your biggest asset.")
            )

        if any(w in msg for w in ["age", "history", "old", "long"]):
            return (
                f"Your credit history length is **{state['history_length']} years**. "
                + ("This is still building — avoid closing your oldest accounts, it only "
                   "improves with time." if state['history_length'] < 3
                   else "This is a solid, established history.")
            )

        if any(w in msg for w in ["inquiry", "inquiries", "apply", "new credit", "loan", "car"]):
            return (
                f"You have **{state['new_credit']}** hard inquiries in the last 6 months. "
                + ("That's on the high side — pausing new applications for a few months lets "
                   "these fade." if state['new_credit'] > 2
                   else "That's a reasonable, low level of recent credit-seeking.")
            )

        if any(w in msg for w in ["target", "gain", "improve", "increase", "raise", "projected"]):
            return (
                f"Based on your current profile, the estimated realistic ceiling is "
                f"**{state['target_score']}** (a potential gain of **+{state['projected_gain']} points** "
                f"from your current {state['score']})."
            )

        if any(w in msg for w in ["score", "status", "overall", "how am i", "summary"]):
            return (
                f"Your current score is **{state['score']}**, rated **{state['status']}**. "
                f"Top recommendation right now: {state['recommendations'][0] if state['recommendations'] else 'Keep up current habits.'}"
            )

        return "Here's what I'd focus on:\n\n" + "\n".join(f"- {r}" for r in state["recommendations"])


agent = EnhancedCreditAgent()

# --- Gradio UI Setup (unchanged) ---
with gr.Blocks(title="AI Credit Score Guidance Agent") as demo:
    gr.Markdown("# 💳 AI Credit Score Guidance Agent")
    gr.Markdown("Adjust your financial profile parameters to trigger the AI agent's analysis, metric comparison, and visualization engine.")

    with gr.Row():
        api_key_in = gr.Textbox(
            label="Google Gemini API Key",
            type="password",
            placeholder="Paste your Google AI Studio key here (starts with AIza...)",
            scale=3,
        )
        connect_btn = gr.Button("🔌 Connect", scale=0, variant="primary")
    status_box = gr.Markdown(check_llm_connection())
    connect_btn.click(fn=connect_llm, inputs=api_key_in, outputs=status_box)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Financial Parameters")
            score_in = gr.Slider(300, 850, value=670, step=1, label="Current Credit Score")
            pay_in = gr.Slider(50, 100, value=92, step=1, label="On-time Payment Record (%)")
            util_in = gr.Slider(0, 100, value=45, step=1, label="Credit Utilization Ratio (%)")
            age_in = gr.Slider(0, 20, value=2, step=0.5, label="Credit History Length (Years)")
            inq_in = gr.Number(value=3, label="Hard Inquiries (Last 6 Months)")
            btn = gr.Button("Run Agent Assessment", variant="primary")

        with gr.Column(scale=1):
            out_summary = gr.Markdown(label="Agent Analysis & Guidance")
            out_plot = gr.Plot(label="Score Trajectory Chart")
            out_df = gr.Dataframe(label="Detailed Metric Breakdown")

    btn.click(
        fn=agent.analyze_profile,
        inputs=[score_in, pay_in, util_in, age_in, inq_in],
        outputs=[out_summary, out_df, out_plot, status_box]
    )

    gr.Markdown("---")
    gr.Markdown("### 💬 Ask the Agent")
    gr.Markdown("Run an assessment above first, then ask follow-up questions — e.g. *\"How can I raise my score?\"* or *\"What about my inquiries?\"*")
    gr.ChatInterface(fn=agent.chat_reply)

demo.launch(share=True, debug=True)
