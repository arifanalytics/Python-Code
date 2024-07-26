# Analyzing file with Google Gemini :

1. Tools yang dipakai : Langchain (library perantara untuk membaca file), Google Gemini (untuk analisa data), FAISSDB (Vector Database yang digunakan saat sesi conversation)
2. Variable tersembunyi : api_key (API KEY Google Gemini yang nantinya dimasukkan dalam database ARIF)
3. Variable yang harus diisi user : summary_length (Panjang summary), iam (Who are you? / profesi user sekarang apa), context (Context dati document), Output (Output Expectation)
4. Bagian pada route "/ask" sebaiknya di loop saja supaya conversation tetap berjalan
5. Hanya bisa menerima file dengan format : ".pdf,.pptx,.csv,.xlsx,.mp3,.docx"
