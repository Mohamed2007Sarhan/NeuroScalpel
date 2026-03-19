package com.neuroscalpel.core;

import com.google.gson.*;
import java.util.*;

/** Parses Phase 1 JSON and manages the ordered queue of EditTasks. */
public class TaskQueue {

    private final List<EditTask> tasks = new ArrayList<>();
    private int currentPos = 0;

    public void reset() { tasks.clear(); currentPos = 0; }
    public int getTotal()           { return tasks.size(); }
    public int getCurrentPosition() { return currentPos; }
    public boolean hasNext()        { return currentPos < tasks.size(); }

    public EditTask popNext() {
        if (!hasNext()) return null;
        EditTask t = tasks.get(currentPos);
        currentPos++;
        return t;
    }

    public List<EditTask> parseFromJson(String rawJson) {
        tasks.clear();
        currentPos = 0;
        try {
            // Strip potential markdown code fences
            String clean = rawJson.replaceAll("(?s)```[a-z]*\\n?", "").strip();
            // Find the first '[' to skip any reasoning preamble
            int start = clean.indexOf('[');
            if (start >= 0) clean = clean.substring(start);
            JsonArray arr = JsonParser.parseString(clean).getAsJsonArray();
            for (int i = 0; i < arr.size(); i++) {
                JsonObject o = arr.get(i).getAsJsonObject();
                tasks.add(new EditTask(
                    i,
                    s(o, "analysis"),
                    s(o, "trick_prompt"),
                    s(o, "subject"),
                    s(o, "wrong_value"),
                    s(o, "correct_value")
                ));
            }
        } catch (Exception e) {
            // Fallback: treat the whole text as a single task
            tasks.add(new EditTask(0, rawJson, rawJson, "unknown", "?", "?"));
        }
        return Collections.unmodifiableList(tasks);
    }

    public String summaryText() {
        var sb = new StringBuilder();
        for (EditTask t : tasks) {
            String mark = (tasks.indexOf(t) < currentPos) ? "[done]" : "[wait]";
            sb.append(mark).append(" ").append(t).append("\n");
        }
        return sb.toString();
    }

    private String s(JsonObject o, String key) {
        return o.has(key) ? o.get(key).getAsString() : "";
    }
}
