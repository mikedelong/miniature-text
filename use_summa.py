from time import time

from summa import summarizer

# todo find a ratio where this works
document = 'Thursday I went to the grocery store. The grocery store is where I buy groceries. I bought groceries there.'
text = """Automatic summarization is the process of reducing a text document with a \
computer program in order to create a summary that retains the most important points \
of the original document. As the problem of information overload has grown, and as \
the quantity of data has increased, so has interest in automatic summarization. \
Technologies that can make a coherent summary take into account variables such as \
length, writing style and syntax. An example of the use of summarization technology \
is search engines such as Google. Document summarization is another."""

text = """must be controlled in real time. Examples include systems \
that interact with humans and robotic systems, such as autonomous \
vehicles. In this paper, we address real-time planning, \
where the planner must return the next action for the \
system to take within a specified wall-clock time bound. \
Providing real-time heuristic search algorithms that are \
complete in domains with dead-end states is a challenging \
problem. Traditional real-time planners are inherently incomplete \
due to the limited time available to make a decision \
even when the state space is fully observable and the actions \
are deterministic. Cserna et al. (2018) proposed the first realtime \
heuristic search method, SafeRTS, that is able to reliably \
reach a goal in domains with dead-ends. Prior real-time \
methods focus their search effort on a single objective that \
minimizes the cost to reach the goal. A single objective is \
insufficient to provide completeness and minimize the time \
to reach the goal as these often contradict each other. Thus, \
SafeRTS distributes the available time between searches optimizing \
the independent objectives of safety and finding the goal. \
The contribution of this work is four-fold. First, we argue \
that benchmark domains used for real-time planning \
may not be good indicators of performance in the context \
of safe real-time planning. We present a new set of benchmarks \
that overcome the deficiencies of previous benchmark \
domains. Second, we show how to utilize meta information \
presented by safety oriented real-time search methods to reduce \
redundant expansions during both the safety and goal-oriented \
searches. This improvement marginally reduces the \
goal achievement time (GAT) while it does not increase \
the space and time complexity of the safe real-time search \
method. Third, we prove inefficiencies in the approach taken \
by SafeRTS by examining properties of local search spaces \
and the changing priority of which nodes to prove safe as \
an LSS grows. Lastly, we introduce a new framework for \
safe real-time heuristic search that utilizes the time bound \
unique to real-time planning. This framework follows the \
same basic principle of search effort distribution as SafeRTS \
but does so more efficiently.We empirically demonstrate the \
potential of the new framework."""

for ratio in range(11):
    float_ratio = float(ratio) / 10.0
    time_before = time()
    summary = summarizer.summarize(ratio=float_ratio, text=text)
    time_after = time()
    print('summarize takes {:5.4f}s for ratio {:5.2f} and summary has length {}'.format(time_after - time_before,
                                                                                        float_ratio, len(summary)))

quit(0)
