import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Container, TextField, Button, Paper, Typography, List, ListItem, ListItemText, Tab, Tabs, CircularProgress, Box, Fade, Slide, Dialog, DialogTitle, DialogContent, DialogActions, Select, MenuItem, AppBar } from '@mui/material';
import Settings from './Settings';
import Portfolio from './Portfolio';
import Login from './Login';
import Register from './Register';
import Dashboard from './Dashboard';
import MarketSentiment from './MarketSentiment';
import InvestmentRecommendation from './InvestmentRecommendation';
import HistoricalDataAnalysis from './HistoricalDataAnalysis';
import UserProfileAnalysis from './UserProfileAnalysis';
import MessageContent from './MessageContent'
import axios from 'axios';
import { logout } from './Auth';
import ErrorBoundary from './ErrorBoundary';

function a11yProps(index) {
    return {
        id: `scrollable-auto-tab-${index}`,
        'aria-controls': `scrollable-auto-tabpanel-${index}`,
    };
}

function App() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([]);
    const [activeTab, setActiveTab] = useState(0);
    const [loading, setLoading] = useState(false);
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [showLogin, setShowLogin] = useState(true);
    const [openBacktest, setOpenBacktest] = useState(false);
    const [backtestResults, setBacktestResults] = useState(null);
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [openComparison, setOpenComparison] = useState(false);
    const [comparisonResults, setComparisonResults] = useState(null);
    const [benchmark, setBenchmark] = useState('SPY');
    const [comparisonStartDate, setComparisonStartDate] = useState('');
    const [comparisonEndDate, setComparisonEndDate] = useState('');
    const [conversationId, setConversationId] = useState(null);
    const [file, setFile] = useState(null);
    const [portfolio, setPortfolio] = useState([]);
    const [portfolioLoading, setPortfolioLoading] = useState(true);
    const [news, setNews] = useState([]);
    const [portfolioLoaded, setPortfolioLoaded] = useState(false);
    const [settings, setSettings] = useState({
        risk_tolerance: 'moderate',
        // ajoutez d'autres paramètres par défaut si nécessaire
    });

    const fetchSettings = async () => {
        try {
            const response = await axios.get(process.env.REACT_APP_API_URL + '/settings');
            setSettings(response.data);
        } catch (error) {
            console.error("Error fetching settings:", error);
        }
    };

    useEffect(() => {
        fetchSettings();
    }, []);


    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            setIsLoggedIn(true);
        }
    }, []);

    const clearChatHistory = useCallback(() => {
        setMessages([]);
        setConversationId(null);
    }, []);

    useEffect(() => {
        // Charger l'historique des chats au montage du composant
        fetchChatHistory();
    }, []);

    const fetchChatHistory = async () => {
        try {
            const response = await axios.get(process.env.REACT_APP_API_URL + '/chat_history');
            setMessages(response.data);
        } catch (error) {
            console.error('Error fetching chat history:', error);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        setLoading(true);
        const newMessage = { role: 'user', content: input };
        setMessages(prevMessages => [...prevMessages, newMessage]);
        setInput('');

        try {
            const response = await axios.post(process.env.REACT_APP_API_URL + '/chat', {
                message: input,
                conversation_id: conversationId
            });
            setConversationId(response.data.conversation_id);
            setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: response.data.reply }]);
        } catch (error) {
            console.error('Error sending message:', error);
        } finally {
            setLoading(false);
        }
    };

    const fetchPortfolio = async () => {
        try {
            const response = await axios.get(process.env.REACT_APP_API_URL + '/portfolio');
            setPortfolio(response.data);
        } catch (error) {
            console.error("Erreur lors de la récupération du portfolio:", error);
        } finally {
            setLoading(false);
        }
    };

    const fetchNews = async () => {
        setLoading(true);
        const tickers = portfolio.map(stock => stock.symbol).join(',');
        try {
            const response = await axios.get(`${process.env.REACT_APP_API_URL}/news?tickers=${tickers}`);
            setNews(response.data);
        } catch (error) {
            console.error("Erreur lors de la récupération des nouvelles:", error);
            setNews([]);
        } finally {
            setLoading(false);
        }
    };

    const fetchLivePrice = useCallback(async (symbol) => {
        if (!symbol) return null;
        try {
            const response = await axios.get(`${process.env.REACT_APP_API_URL}/live_price?symbol=${symbol}`);
            return response.data.price;
        } catch (error) {
            console.error(`Error fetching live price for ${symbol}:`, error);
            return null;
        }
    }, []);

    useEffect(() => {
        fetchLivePrice();
    }, [fetchLivePrice]);

    useEffect(() => {
        if (portfolio.length > 0) {
            fetchNews();
        }
    }, [portfolio]);

    const handleAgentCall = async (agentName) => {
        setLoading(true);
        let data = {};
        switch (agentName) {
            case 'document':
                data = { text: input };
                break;
            case 'sentiment':
                data = { company: input };
                break;
            case 'financial_modeling':
                data = { ticker: input };
                break;
            case 'portfolio_optimization':
            case 'risk_management':
                data = { tickers: input.split(','), portfolio_value: 100000 };
                break;
            case 'reporting':
                data = { portfolio_data: portfolio.stocks };
            case 'compliance':
                try {
                    // Essayez de parser input comme JSON
                    const parsedInput = JSON.parse(input);
                    data = { portfolio_data: parsedInput };
                } catch (error) {
                    // Si le parsing échoue, supposons que c'est une chaîne de caractères représentant un portefeuille
                    console.warn("Input is not valid JSON, treating it as a string representation of portfolio");
                    // Vous pouvez implémenter ici une logique pour convertir la chaîne en objet de portfolio
                    // Par exemple, si la chaîne est au format "AAPL:30%,GOOGL:40%,MSFT:30%"
                    const portfolioArray = input.split(',').map(item => {
                        const [symbol, weight] = item.split(':');
                        return { symbol, weight: parseFloat(weight) };
                    });
                    data = { portfolio_data: portfolioArray };
                }
                break;
            case 'market_sentiment':
                data = { ticker: input };
                break;
            case 'user_profile_analysis':
                // Nous allons utiliser l'ID de l'utilisateur connecté
                data = { user_id: localStorage.getItem('userId') };
                break;
            case 'historical_data_analysis':
                // Supposons que l'utilisateur entre le ticker dans l'input
                data = {
                    ticker: input,
                    start_date: startDate,  // Assurez-vous que ces variables sont définies dans votre composant
                    end_date: endDate
                };
                break;
            case 'investment_recommendation':
                try {
                    const settingsResponse = await axios.get(process.env.REACT_APP_API_URL + '/settings');
                    data = {
                        portfolio: portfolio.stocks ? portfolio.stocks.map(stock => stock.symbol) : [],
                        risk_profile: settingsResponse.data.risk_profile || 'moderate'
                    };
                } catch (error) {
                    console.error("Error fetching settings:", error);
                    data = {
                        portfolio: portfolio.stocks ? portfolio.stocks.map(stock => stock.symbol) : [],
                        risk_profile: 'moderate'
                    };
                }
                break;
            default:
                data = { input: input };
                break;
        }

        try {
            const response = await axios.post(`${process.env.REACT_APP_API_URL}/agent/${agentName}`, data);
            let newMessage;
            if (agentName === 'reporting') {
                // Traitement spécial pour l'agent de reporting
                newMessage = {
                    role: 'assistant',
                    content: response.data.content,
                    graphs: response.data.graph
                };
            } else {
                // Traitement standard pour les autres agents
                newMessage = {
                    role: 'assistant',
                    content: JSON.stringify(response.data, null, 2),
                    graph: response.data.graph
                };
            }
            setMessages(prevMessages => [...prevMessages, newMessage]);

            // Sauvegarder le message dans l'historique du chat
            await axios.post(process.env.REACT_APP_API_URL + '/chat_history', newMessage);
        } catch (error) {
            console.error(`Error calling ${agentName} agent:`, error);
            const errorMessage = {
                role: 'assistant',
                content: `Error: ${error.response?.data?.error || error.message}`
            };
            setMessages(prevMessages => [...prevMessages, errorMessage]);

            // Sauvegarder le message d'erreur dans l'historique du chat
            await axios.post(process.env.REACT_APP_API_URL + '/chat_history', errorMessage);
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (file && file.type === 'application/pdf') {
            setFile(file);
            setLoading(true);
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await axios.post(process.env.REACT_APP_API_URL + '/upload_pdf', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: JSON.stringify(response.data, null, 2) }]);
            } catch (error) {
                console.error('Error uploading PDF:', error);
            } finally {
                setLoading(false);
            }
        }
    };
    const handlePDFUpload = async (e) => {
        const file = e.target.files[0];
        if (file && file.type === 'application/pdf') {
            setLoading(true);
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await axios.post(process.env.REACT_APP_API_URL + '/upload_pdf', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: JSON.stringify(response.data, null, 2) }]);
            } catch (error) {
                console.error('Error uploading PDF:', error);
            } finally {
                setLoading(false);
            }
        }
    };

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            setIsLoggedIn(true);
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            fetchPortfolio();
        } else {
            setIsLoggedIn(false);
        }
    }, []);

    const handleDragOver = (e) => {
        e.preventDefault();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/pdf') {
            handlePDFUpload({ target: { files: [file] } });
        }
    };

    // Assurez-vous d'inclure le token dans toutes les requêtes
    axios.interceptors.request.use(
        (config) => {
            const token = localStorage.getItem('token');
            if (token) {
                config.headers['Authorization'] = `Bearer ${token}`;
            }
            return config;
        },
        (error) => {
            return Promise.reject(error);
        }
    );

    const handleLogin = () => {
        setIsLoggedIn(true);
        // Assurez-vous que le token est stocké dans localStorage
        const token = localStorage.getItem('token');
        if (token) {
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        }
    };

    const handleLogout = () => {
        logout();
        setIsLoggedIn(false);
    };

    const runBacktest = async () => {
        if (!portfolio || !portfolio.stocks || portfolio.stocks.length === 0) {
            alert("Portfolio is empty. Please add some stocks before running the backtest.");
            return;
        }

        try {
            const response = await axios.post(process.env.REACT_APP_API_URL + '/backtest', {
                portfolio: portfolio,
                start_date: startDate,
                end_date: endDate
            });
            setBacktestResults(response.data);
            setOpenBacktest(true);
        } catch (error) {
            console.error("Error running backtest:", error);
            if (error.response && error.response.data && error.response.data.error) {
                alert(`Error running backtest: ${error.response.data.error}`);
            } else {
                alert('Error running backtest. Please try again.');
            }
        }
    };

    const compareWithBenchmark = async () => {
        try {
            const response = await axios.post(process.env.REACT_APP_API_URL + '/compare_portfolios', {
                portfolio: portfolio.stocks, // Assurez-vous que c'est bien un tableau d'objets stock
                benchmark: benchmark,
                start_date: comparisonStartDate,
                end_date: comparisonEndDate
            });
            setComparisonResults(response.data);
            setOpenComparison(true);
        } catch (error) {
            console.error("Error comparing portfolios:", error);
            if (error.response) {
                console.error("Response data:", error.response.data);
                console.error("Response status:", error.response.status);
            }
            alert('Error comparing portfolios. Please try again.');
        }
    };

    const onClearChat = useCallback(async () => {
        try {
            await axios.post(process.env.REACT_APP_API_URL + '/clear_chat');
            setMessages([]);
            setConversationId(null);
            alert('Chat history cleared successfully');
        } catch (error) {
            console.error('Error clearing chat history:', error);
            alert('Error clearing chat history. Please try again.');
        }
    }, []);

    if (!isLoggedIn) {
        return (
            <Container maxWidth="sm">
                <Box mt={4}>
                    {showLogin ? (
                        <>
                            <Login onLogin={handleLogin} />
                            <Button onClick={() => setShowLogin(false)}>
                                Don't have an account? Register
                            </Button>
                        </>
                    ) : (
                        <>
                            <Register onRegisterSuccess={() => setShowLogin(true)} />
                            <Button onClick={() => setShowLogin(true)}>
                                Already have an account? Login
                            </Button>
                        </>
                    )}
                </Box>
            </Container>
        );
    }
    return (
        <Fade in={true} timeout={1000}>
            <Slide direction="up" in={true} mountOnEnter unmountOnExit>
                <Container maxWidth="md">
                    <Paper elevation={3} style={{ padding: '20px', marginTop: '20px' }}>
                        <Button onClick={handleLogout}>Logout</Button>
                        <Typography variant="h4" gutterBottom>
                            AI Copilot
                        </Typography>
                        <AppBar position="static" color="default">
                            <Tabs
                                value={activeTab}
                                onChange={(e, newValue) => setActiveTab(newValue)}
                                indicatorColor="primary"
                                textColor="primary"
                                variant="scrollable"
                                scrollButtons="auto"
                                aria-label="scrollable auto tabs example"
                            >
                                <Tab label="Copilot" {...a11yProps(0)} />
                                <Tab label="Agents" {...a11yProps(1)} />
                                <Tab label="PDF Analysis" {...a11yProps(2)} />
                                <Tab label="Settings" {...a11yProps(3)} />
                                <Tab label="Portfolio" {...a11yProps(4)} />
                                <Tab label="Chargement Portfolio" {...a11yProps(5)} />
                                <Tab label="Market Sentiment" {...a11yProps(6)} />
                                <Tab label="Investment Recommendation" {...a11yProps(7)} />
                                <Tab label="Historical Data Analysis" {...a11yProps(8)} />
                                <Tab label="User Profile Analysis" {...a11yProps(9)} />
                            </Tabs>
                        </AppBar>
                        <Box p={3}>
                            {activeTab === 0 && (
                                <>
                                    <List>
                                        {messages.map((message, index) => (
                                            <ListItem key={index} alignItems="flex-start">
                                                <ListItemText
                                                    primary={message.role === 'user' ? 'You' : 'AI'}
                                                    secondary={<MessageContent content={message.content} />}
                                                />
                                            </ListItem>
                                        ))}
                                    </List>
                                    <form onSubmit={handleSubmit}>
                                        <TextField
                                            fullWidth
                                            variant="outlined"
                                            value={input}
                                            onChange={(e) => setInput(e.target.value)}
                                            placeholder="Type your message..."
                                            margin="normal"
                                        />
                                        <Button type="submit" variant="contained" color="primary" disabled={loading}>
                                            {loading ? <CircularProgress size={24} /> : 'Send'}
                                        </Button>
                                    </form>
                                    <input
                                        type="file"
                                        accept=".pdf"
                                        onChange={handlePDFUpload}
                                        style={{ display: 'none' }}
                                        id="pdf-upload"
                                    />
                                    <label htmlFor="pdf-upload">
                                        <Button variant="contained" component="span" color="primary">
                                            Upload PDF
                                        </Button>
                                    </label>
                                    <Typography variant="body2" style={{ marginTop: '10px' }}>
                                        You can also drag and drop a PDF file here
                                    </Typography>
                                </>
                            )}
                            {activeTab === 1 && (
                                <>
                                    <TextField
                                        fullWidth
                                        variant="outlined"
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        placeholder="Enter data for agent..."
                                        margin="normal"
                                    />
                                    <Button onClick={() => handleAgentCall('document')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Document Analysis
                                    </Button>
                                    <Button onClick={() => handleAgentCall('sentiment')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Sentiment Analysis
                                    </Button>
                                    <Button onClick={() => handleAgentCall('financial_modeling')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Financial Modeling
                                    </Button>
                                    <Button onClick={() => handleAgentCall('portfolio_optimization')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Portfolio Optimization
                                    </Button>
                                    <Button onClick={() => handleAgentCall('risk_management')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Risk Management
                                    </Button>
                                    <Button onClick={() => handleAgentCall('reporting')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Reporting
                                    </Button>
                                    <Button onClick={() => handleAgentCall('compliance')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Compliance Check
                                    </Button>
                                    <Button onClick={() => handleAgentCall('market_sentiment')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Market Sentiment
                                    </Button>
                                    <Button onClick={() => handleAgentCall('user_profile_analysis')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        User Profile Analysis
                                    </Button>
                                    <Button onClick={() => handleAgentCall('historical_data_analysis')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Historical Data Analysis
                                    </Button>
                                    <Button onClick={() => handleAgentCall('investment_recommendation')} variant="contained" color="primary" style={{ margin: '5px' }} disabled={loading}>
                                        Investment Recommendation
                                    </Button>
                                    <List>
                                        {messages.map((message, index) => (
                                            <ListItem key={index} alignItems="flex-start">
                                                <ListItemText
                                                    primary={message.role === 'user' ? 'You' : 'AI'}
                                                    secondary={
                                                        <>
                                                            <MessageContent content={message.content} />
                                                            {message.graphs && message.graphs.map((graph, graphIndex) => (
                                                                <img
                                                                    key={graphIndex}
                                                                    src={`data:image/png;base64,${graph}`}
                                                                    alt={`Portfolio Graph ${graphIndex + 1}`}
                                                                    style={{ maxWidth: '100%', marginTop: '10px' }}
                                                                />
                                                            ))}
                                                            {message.graph && (
                                                                <img
                                                                    src={`data:image/png;base64,${message.graph}`}
                                                                    alt="Graph"
                                                                    style={{ maxWidth: '100%', marginTop: '10px' }}
                                                                />
                                                            )}
                                                        </>
                                                    }
                                                />
                                            </ListItem>
                                        ))}
                                    </List>
                                </>
                            )}
                            {activeTab === 2 && (
                                <>
                                    <input
                                        type="file"
                                        accept=".pdf"
                                        onChange={handleFileUpload}
                                        style={{ display: 'none' }}
                                        id="pdf-upload"
                                    />
                                    <label htmlFor="pdf-upload">
                                        <Button variant="contained" component="span" color="primary" disabled={loading}>
                                            Upload PDF
                                        </Button>
                                    </label>
                                    {file && <Typography variant="body1">{file.name}</Typography>}
                                    {loading && <CircularProgress />}
                                    <List>
                                        {messages.map((message, index) => (
                                            <ListItem key={index} alignItems="flex-start">
                                                <ListItemText
                                                    primary={message.role === 'user' ? 'You' : 'AI'}
                                                    secondary={<MessageContent content={message.content} />}
                                                />
                                            </ListItem>
                                        ))}
                                    </List>
                                </>
                            )}
                            {activeTab === 3 && <Settings onClearChat={clearChatHistory} />}
                            {activeTab === 4 && <Portfolio />}

                            <Button onClick={() => setOpenBacktest(true)}>Run Backtest</Button>

                            <Dialog
                                open={openBacktest}
                                onClose={() => setOpenBacktest(false)}
                                fullWidth
                                maxWidth="lg"
                                PaperProps={{
                                    style: {
                                        width: '65%',
                                        height: '73%',
                                        maxWidth: 'none',
                                        maxHeight: 'none',
                                        margin: 'auto'
                                    }
                                }}
                            >
                                <DialogTitle>Backtest Results</DialogTitle>
                                <DialogContent>
                                    <TextField
                                        label="Start Date"
                                        type="date"
                                        value={startDate}
                                        onChange={(e) => setStartDate(e.target.value)}
                                        InputLabelProps={{ shrink: true }}
                                        fullWidth
                                        margin="normal"
                                    />
                                    <TextField
                                        label="End Date"
                                        type="date"
                                        value={endDate}
                                        onChange={(e) => setEndDate(e.target.value)}
                                        InputLabelProps={{ shrink: true }}
                                        fullWidth
                                        margin="normal"
                                    />
                                    <Button onClick={runBacktest}>Run Backtest</Button>
                                    {backtestResults && (
                                        <>
                                            <p>Total Return: {(backtestResults.total_return * 100).toFixed(2)}%</p>
                                            <p>Annualized Return: {(backtestResults.annualized_return * 100).toFixed(2)}%</p>
                                            <p>Volatility: {(backtestResults.volatility * 100).toFixed(2)}%</p>
                                            <p>Sharpe Ratio: {backtestResults.sharpe_ratio.toFixed(2)}</p>
                                            <div style={{ width: '100%', height: '41%' }}>
                                                <ResponsiveContainer>
                                                    <LineChart width="100%" height={800} data={backtestResults.portfolio_values.map((value, index) => ({ date: index, value }))}>
                                                        <XAxis dataKey="date" />
                                                        <YAxis />
                                                        <CartesianGrid strokeDasharray="3 3" />
                                                        <Tooltip />
                                                        <Legend />
                                                        <Line type="monotone" dataKey="value" stroke="#8884d8" />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </>
                                    )}
                                </DialogContent>
                                <DialogActions>
                                    <Button onClick={() => setOpenBacktest(false)}>Close</Button>
                                </DialogActions>
                            </Dialog>
                            <Button onClick={() => setOpenComparison(true)}>Compare with Benchmark</Button>

                            <Dialog open={openComparison} onClose={() => setOpenComparison(false)}>
                                <DialogTitle>Portfolio Comparison</DialogTitle>
                                <DialogContent>
                                    <Select
                                        value={benchmark}
                                        onChange={(e) => setBenchmark(e.target.value)}
                                        fullWidth
                                        margin="normal"
                                    >
                                        <MenuItem value="SPY">S&P 500 (SPY)</MenuItem>
                                        <MenuItem value="QQQ">Nasdaq 100 (QQQ)</MenuItem>
                                        <MenuItem value="IWM">Russell 2000 (IWM)</MenuItem>
                                    </Select>
                                    <TextField
                                        label="Start Date"
                                        type="date"
                                        value={comparisonStartDate}
                                        onChange={(e) => setComparisonStartDate(e.target.value)}
                                        InputLabelProps={{ shrink: true }}
                                        fullWidth
                                        margin="normal"
                                    />
                                    <TextField
                                        label="End Date"
                                        type="date"
                                        value={comparisonEndDate}
                                        onChange={(e) => setComparisonEndDate(e.target.value)}
                                        InputLabelProps={{ shrink: true }}
                                        fullWidth
                                        margin="normal"
                                    />
                                    {/* <Button onClick={compareWithBenchmark}>Compare</Button> */}
                                    {comparisonResults && (
                                        <>
                                            <p>Portfolio Return: {(comparisonResults.portfolio_return * 100).toFixed(2)}%</p>
                                            <p>Benchmark Return: {(comparisonResults.benchmark_return * 100).toFixed(2)}%</p>
                                            <p>Portfolio Volatility: {(comparisonResults.portfolio_volatility * 100).toFixed(2)}%</p>
                                            <p>Benchmark Volatility: {(comparisonResults.benchmark_volatility * 100).toFixed(2)}%</p>
                                            <p>Portfolio Sharpe Ratio: {comparisonResults.portfolio_sharpe?.toFixed(2) || 'N/A'}</p>
                                            <p>Benchmark Sharpe Ratio: {comparisonResults.benchmark_sharpe?.toFixed(2) || 'N/A'}</p>
                                            {comparisonResults.portfolio_cumulative && comparisonResults.benchmark_cumulative && (
                                                <LineChart width={500} height={300} data={comparisonResults.portfolio_cumulative.map((value, index) => ({
                                                    date: index,
                                                    portfolio: value,
                                                    benchmark: comparisonResults.benchmark_cumulative[index]
                                                }))}>
                                                    <XAxis dataKey="date" />
                                                    <YAxis />
                                                    <CartesianGrid strokeDasharray="3 3" />
                                                    <Tooltip />
                                                    <Legend />
                                                    <Line type="monotone" dataKey="portfolio" stroke="#8884d8" />
                                                    <Line type="monotone" dataKey="benchmark" stroke="#82ca9d" />
                                                </LineChart>
                                            )}
                                        </>
                                    )}
                                </DialogContent>
                                <DialogActions>
                                    <Button onClick={compareWithBenchmark}>Compare</Button>
                                    <Button onClick={() => setOpenComparison(false)}>Close</Button>
                                </DialogActions>
                            </Dialog>
                            {activeTab === 5 && (
                                <ErrorBoundary>
                                    {portfolioLoading ? (
                                        <Typography>Chargement du portfolio...</Typography>
                                    ) : portfolio ? (
                                        <Dashboard portfolio={portfolio} />
                                    ) : (
                                        <Typography>Erreur lors du chargement du portfolio</Typography>
                                    )}
                                </ErrorBoundary>
                            )}
                            {activeTab === 6 && <MarketSentiment />}
                            {activeTab === 7 && <InvestmentRecommendation />}
                            {activeTab === 8 && <HistoricalDataAnalysis />}
                            {activeTab === 9 && <UserProfileAnalysis />}
                        </Box>
                    </Paper>
                </Container>
            </Slide >
        </Fade >
    );
}

export default App;