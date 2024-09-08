package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/rs/cors"
)

type IdeaRequest struct {
	Domain      string `json:"domain"`
	Description string `json:"description"`
}

type Idea struct {
	Name     string `json:"name"`
	Concept  string `json:"concept"`
	Features string `json:"features"`
}

type IdeaResponse struct {
	Ideas []Idea `json:"ideas"`
}

type GroqClient struct {
	ApiKey string
}

type GroqMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type GroqRequest struct {
	Model       string        `json:"model"`
	Messages    []GroqMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
	TopP        float64       `json:"top_p"`
	Stream      bool          `json:"stream"`
	Stop        any           `json:"stop"`
}

func main() {

	_ = godotenv.Load()

	http.HandleFunc("/api/generate-ideas", generateIdeasHandler)

	allowedOrigins := strings.Split(os.Getenv("ALLOWED_ORIGINS"), ",")
	if len(allowedOrigins) == 0 || (len(allowedOrigins) == 1 && allowedOrigins[0] == "") {
		allowedOrigins = []string{"http://localhost:3000"} // Fallback for local development
	}

	c := cors.New(cors.Options{
		AllowedOrigins:   allowedOrigins,
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Access-Control-Allow-Headers", "Content-Type", "Authorization"},
		AllowCredentials: true,
		Debug:            true, // Enable for debugging, remove in production
	})

	// Wrap your handlers with the CORS middleware
	handler := c.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate-ideas" {
			generateIdeasHandler(w, r)
			return
		}
		http.NotFound(w, r)
	}))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("Server is running on port %s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, handler))
}

func generateIdeasHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req IdeaRequest
	err := json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ideas, err := generateIdeas(req.Domain, req.Description)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := IdeaResponse{Ideas: ideas}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func generateIdeas(domain, description string) ([]Idea, error) {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GROQ_API_KEY not set")
	}

	groqReq := GroqRequest{
		Model: "llama3-8b-8192",
		Messages: []GroqMessage{
			{
				Role:    "system",
				Content: "You are an AI assistant that generates project ideas. Your output must be a valid JSON array of objects, each with exactly three fields: 'name', 'concept', and 'features'. The 'features' field must be a single string with comma-separated values. Do not include any explanation or additional text. Generate exactly 5 ideas based on this format: [{'name': 'Project Name', 'concept': 'Short description', 'features': 'Feature 1, Feature 2, Feature 3'}]. Ensure the JSON array is properly closed with a square bracket ']' at the end.",
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("Generate 3 project ideas for the domain: %s. Description: %s", domain, description),
			},
		},
		Temperature: 0.7,
		MaxTokens:   1240,
		TopP:        1,
		Stream:      false,
		Stop:        nil,
	}

	jsonData, err := json.Marshal(groqReq)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", "https://api.groq.com/openai/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	err = json.Unmarshal(body, &result)
	if err != nil {
		return nil, err
	}

	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil, fmt.Errorf("unexpected response format")
	}

	firstChoice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected choice format")
	}

	message, ok := firstChoice["message"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected message format")
	}

	content, ok := message["content"].(string)
	if !ok {
		return nil, fmt.Errorf("unexpected content format")
	}

	return parseIdeas(content)
}

func parseIdeas(content string) ([]Idea, error) {
	var ideas []Idea
	err := json.Unmarshal([]byte(content), &ideas)
	if err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %v", err)
	}

	if len(ideas) != 3 {
		return nil, fmt.Errorf("expected 3 ideas, got %d", len(ideas))
	}

	for _, idea := range ideas {
		if idea.Name == "" || idea.Concept == "" || idea.Features == "" {
			return nil, fmt.Errorf("invalid idea format: all fields must be non-empty")
		}
	}

	return ideas, nil
}
