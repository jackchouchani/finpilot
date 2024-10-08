import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ThemeProvider, createTheme, styled, alpha } from '@mui/material/styles';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import {
    Container, TextField, Button, Paper, Typography, List, ListItem, ListItemText,
    Tab, Tabs, CircularProgress, Box, Fade, Slide, Dialog, DialogTitle,
    DialogContent, DialogActions, Select, MenuItem, AppBar, Switch,
    FormControlLabel, Toolbar, IconButton, Drawer, ListItemIcon, InputBase,
    Badge, Menu, useMediaQuery, CssBaseline, Stack
} from '@mui/material';
import {
    Menu as MenuIcon,
    Dashboard as DashboardIcon,
    AccountBalance as AccountBalanceIcon,
    TrendingUp as TrendingUpIcon,
    Settings as SettingsIcon,
    Logout as LogoutIcon,
    Chat as ChatIcon,
    GroupWork as GroupWorkIcon,
    PictureAsPdf as PictureAsPdfIcon,
    AccountCircle,
    Notifications as NotificationsIcon,
    MonetizationOn as MonetizationOnIcon,
    History as HistoryIcon,
    Person as PersonIcon
} from '@mui/icons-material';
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
import ChatBox from './ChatBox';

const api = axios.create({
    baseURL: process.env.REACT_APP_API_URL,
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    },
    withCredentials: true
});

api.interceptors.request.use(
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

const drawerWidth = 240;

function App() {
    const [darkMode, setDarkMode] = useState(false);
    const [open, setOpen] = useState(true);
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
    const [settings, setSettings] = useState({
        risk_tolerance: 'moderate',
    });
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const agentInputRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const toggleDarkMode = () => {
        setDarkMode(!darkMode);
    };

    const theme = createTheme({
        palette: {
            primary: {
                main: '#1976d2',
                light: '#64b5f6',
                dark: '#0d47a1',
            },
            secondary: {
                main: '#2196f3',
            },
            background: {
                default: '#f5f5f5',
                paper: '#ffffff',
            },
            text: {
                primary: '#212121',
            },
        },
        typography: {
            fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
            h4: {
                fontWeight: 600,
            },
        },
        components: {
            MuiButton: {
                styleOverrides: {
                    root: {
                        textTransform: 'none',
                    },
                },
            },
            MuiPaper: {
                styleOverrides: {
                    root: {
                        borderRadius: 8,
                    },
                },
            },
        },
    });

    const fetchSettings = async () => {
        try {
            const response = await axios.get(
                `${process.env.REACT_APP_API_URL}/settings`,
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                }
            );
            setSettings(response.data);
        } catch (error) {
            console.error('Erreur lors de la récupération des paramètres:', error);
            if (error.response) {
                console.error('Données de réponse:', error.response.data);
            }
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
        const fetchChatHistory = async () => {
            try {
                const response = await axios.get(`${process.env.REACT_APP_API_URL}/chat_history`, {
                    headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
                });
                setMessages(response.data.reverse());
            } catch (error) {
                console.error('Error fetching chat history:', error);
            }
        };

        if (isLoggedIn) {
            fetchChatHistory();
        }
    }, [isLoggedIn]);

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
        console.log("Agent Input Ref:", agentInputRef); // Ajoutez cette ligne
        console.log("Agent Input Ref Current:", agentInputRef.current); // Ajoutez cette ligne
        const inputValue = agentInputRef.current ? agentInputRef.current.value : '';
        console.log("Input value:", inputValue);
        let data = {};
        switch (agentName) {
            case 'document':
                data = { text: inputValue };
                break;
            case 'sentiment':
                data = { company: inputValue };
                break;
            case 'financial_modeling':
                data = { ticker: inputValue };
                break;
            case 'portfolio_optimization':
            case 'risk_management':
                data = { tickers: inputValue.split(','), portfolio_value: 100000 };
                break;
            case 'reporting':
                data = { portfolio_data: portfolio.stocks };
                break;
            case 'compliance':
                try {
                    const parsedInput = JSON.parse(inputValue);
                    data = { portfolio_data: parsedInput };
                } catch (error) {
                    console.warn("Input is not valid JSON, treating it as a string representation of portfolio");
                    const portfolioArray = inputValue.split(',').map(item => {
                        const [symbol, weight] = item.split(':');
                        return { symbol, weight: parseFloat(weight) };
                    });
                    data = { portfolio_data: portfolioArray };
                }
                break;
            case 'market_sentiment':
                data = { ticker: inputValue };
                break;
            case 'user_profile_analysis':
                data = { user_id: localStorage.getItem('userId') };
                break;
            case 'historical_data_analysis':
                data = {
                    ticker: inputValue,
                    start_date: startDate,
                    end_date: endDate
                };
                break;
            case 'investment_recommendation':
                data = {
                    portfolio: portfolio.stocks ? portfolio.stocks.map(stock => stock.symbol) : [],
                    risk_profile: settings.risk_profile || 'moderate'
                };
                break;
            default:
                data = { input: inputValue };
                break;
        }
        const userMessage = { role: 'user', content: inputValue };
        setMessages(prevMessages => [userMessage, ...prevMessages]);
        console.log("Data being sent:", data); // Ajoutez cette ligne
        try {
            await axios.post(process.env.REACT_APP_API_URL + '/chat_history', userMessage);
        } catch (error) {
            console.error('Error saving user message to chat history:', error);
        }

        try {
            const response = await axios.post(`${process.env.REACT_APP_API_URL}/agent/${agentName}`, data);
            let newMessage = {
                role: 'assistant',
                content: typeof response.data === 'string' ? response.data : response.data.content || 'Pas de contenu reçu de l\'agent'
            };
            if (response.data.graphs) {
                newMessage.graphs = response.data.graphs;
            }
            setMessages(prevMessages => [...prevMessages, newMessage]);

            try {
                await axios.post(process.env.REACT_APP_API_URL + '/chat_history', newMessage);
            } catch (saveError) {
                console.error('Error saving agent message to chat history:', saveError);
            }
        } catch (error) {
            console.error(`Error calling ${agentName} agent:`, error);
            const errorMessage = {
                role: 'assistant',
                content: `Error: ${error.response?.data?.error || error.message}`
            };
            setMessages(prevMessages => [...prevMessages, errorMessage]);

            try {
                await axios.post(process.env.REACT_APP_API_URL + '/chat_history', errorMessage);
            } catch (saveError) {
                console.error('Error saving error message to chat history:', saveError);
            }
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

    const handleSubmit = async (e, inputValue) => {
        e.preventDefault();
        if (!inputValue) return;

        setLoading(true);
        const newMessage = { role: 'user', content: inputValue };
        setMessages(prevMessages => [...prevMessages, newMessage]);

        try {
            // Récupérer le portfolio de l'utilisateur
            const portfolioResponse = await axios.get(`${process.env.REACT_APP_API_URL}/portfolio`);
            const userPortfolio = portfolioResponse.data;

            // Préparer les données à envoyer
            const requestData = {
                message: inputValue,
                conversation_id: conversationId,
                portfolio: userPortfolio,
                risk_profile: settings.riskProfile, // Assurez-vous que cette information est disponible dans votre state
            };

            const response = await axios.post(`${process.env.REACT_APP_API_URL}/chat`, requestData, {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (response.data && response.data.reply) {
                setConversationId(response.data.conversation_id);
                setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: response.data.reply }]);
            } else {
                throw new Error('Réponse invalide du serveur');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: 'Désolé, une erreur est survenue. Veuillez réessayer.' }]);
        } finally {
            setLoading(false);
        }
    };

    if (!isLoggedIn) {
        return (
            <ThemeProvider theme={theme}>
                <CssBaseline />
                <Container maxWidth="sm">
                    <Box mt={4}>
                        {showLogin ? (
                            <>
                                <Login onLogin={handleLogin} />
                                <Button onClick={() => setShowLogin(false)}>
                                    Pas de compte ? S'inscrire
                                </Button>
                            </>
                        ) : (
                            <>
                                <Register onRegisterSuccess={() => setShowLogin(true)} />
                                <Button onClick={() => setShowLogin(true)}>
                                    Déjà un compte ? Se connecter
                                </Button>
                            </>
                        )}
                    </Box>
                </Container>
            </ThemeProvider>
        );
    }

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <ResponsiveAppContent
                darkMode={darkMode}
                setDarkMode={setDarkMode}
                drawerOpen={open}
                setDrawerOpen={setOpen}
                // Passez toutes les autres props nécessaires ici
                handleLogout={handleLogout}
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                messages={messages}
                handleSubmit={handleSubmit}
                loading={loading}
                handlePDFUpload={handlePDFUpload}
                handleAgentCall={handleAgentCall}
                handleFileUpload={handleFileUpload}
                file={file}
                clearChatHistory={clearChatHistory}
                portfolio={portfolio}
                portfolioLoading={portfolioLoading}
                openBacktest={openBacktest}
                setOpenBacktest={setOpenBacktest}
                openComparison={openComparison}
                setOpenComparison={setOpenComparison}
                runBacktest={runBacktest}
                compareWithBenchmark={compareWithBenchmark}
                startDate={startDate}
                setStartDate={setStartDate}
                endDate={endDate}
                setEndDate={setEndDate}
                benchmark={benchmark}
                setBenchmark={setBenchmark}
                comparisonStartDate={comparisonStartDate}
                setComparisonStartDate={setComparisonStartDate}
                comparisonEndDate={comparisonEndDate}
                setComparisonEndDate={setComparisonEndDate}
                backtestResults={backtestResults}
                comparisonResults={comparisonResults}
                messagesEndRef={messagesEndRef}
                inputRef={inputRef}
                agentInputRef={agentInputRef}
            />
        </ThemeProvider>
    );
}

// Nouveau composant qui gère la requête média
function ResponsiveAppContent(props) {
    const isMobile = useMediaQuery((theme) => theme.breakpoints.down('sm'));

    return <AppContent {...props} isMobile={isMobile} />;
}

function AppContent({
    darkMode,
    setDarkMode,
    drawerOpen,
    setDrawerOpen,
    handleLogout,
    activeTab,
    setActiveTab,
    messages,
    handleSubmit,
    loading,
    handlePDFUpload,
    handleAgentCall,
    handleFileUpload,
    file,
    clearChatHistory,
    portfolio,
    portfolioLoading,
    openBacktest,
    setOpenBacktest,
    openComparison,
    setOpenComparison,
    runBacktest,
    compareWithBenchmark,
    startDate,
    setStartDate,
    endDate,
    setEndDate,
    benchmark,
    setBenchmark,
    comparisonStartDate,
    setComparisonStartDate,
    comparisonEndDate,
    setComparisonEndDate,
    backtestResults,
    comparisonResults,
    isMobile,
    messagesEndRef,
    inputRef,
    agentInputRef,
}) {
    const [anchorEl, setAnchorEl] = React.useState(null);

    const handleMenu = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    return (
        <Box sx={{ display: 'flex' }}>
            <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
                <Toolbar>
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        edge="start"
                        onClick={() => setDrawerOpen(!drawerOpen)}
                        sx={{ marginRight: 2, ...(drawerOpen && { display: 'none' }) }}
                    >
                        <MenuIcon />
                    </IconButton>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <img src="./logo.jpg" alt="FinPilot Logo" style={{ height: 50, marginRight: 16 }} />
                        <Typography variant="h6" noWrap component="div">
                            FinPilot
                        </Typography>
                    </Box>
                    <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
                        <IconButton color="inherit">
                            <Badge badgeContent={4} color="secondary">
                                <NotificationsIcon />
                            </Badge>
                        </IconButton>
                        <IconButton
                            edge="end"
                            aria-label="account of current user"
                            aria-haspopup="true"
                            onClick={handleMenu}
                            color="inherit"
                        >
                            <AccountCircle />
                        </IconButton>
                    </Box>
                    <FormControlLabel
                        control={<Switch checked={darkMode} onChange={() => setDarkMode(!darkMode)} />}
                        label="Mode Sombre"
                        sx={{ ml: 2 }}
                    />
                </Toolbar>
            </AppBar>
            <Drawer
                variant={isMobile ? "temporary" : "permanent"}
                open={isMobile ? drawerOpen : true}
                onClose={() => setDrawerOpen(false)}
                sx={{
                    width: drawerWidth,
                    flexShrink: 0,
                    [`& .MuiDrawer-paper`]: {
                        width: drawerWidth,
                        boxSizing: 'border-box',
                        backgroundColor: 'background.default',
                        borderRight: '1px solid',
                        borderColor: 'divider',
                    },
                }}
            >
                <Toolbar />
                <Box sx={{ overflow: 'auto' }}>
                    <List>
                        {[
                            { text: 'Copilote', icon: <ChatIcon /> },
                            { text: 'Agents', icon: <GroupWorkIcon /> },
                            { text: 'Analyse PDF', icon: <PictureAsPdfIcon /> },
                            { text: 'Paramètres', icon: <SettingsIcon /> },
                            { text: 'Portefeuille', icon: <AccountBalanceIcon /> },
                            { text: 'Sentiment du Marché', icon: <TrendingUpIcon /> },
                            { text: 'Recommandation d\'Investissement', icon: <MonetizationOnIcon /> },
                            { text: 'Analyse de Données Historiques', icon: <HistoryIcon /> },
                            { text: 'Analyse du Profil Utilisateur', icon: <PersonIcon /> }
                        ].map((item, index) => (
                            <ListItem
                                button
                                key={item.text}
                                onClick={() => setActiveTab(index)}
                                selected={activeTab === index}
                                sx={{
                                    '&.Mui-selected': {
                                        backgroundColor: 'primary.light',
                                        '&:hover': {
                                            backgroundColor: 'primary.light',
                                        },
                                    },
                                    '&:hover': {
                                        backgroundColor: 'action.hover',
                                    },
                                }}
                            >
                                <ListItemIcon sx={{ color: activeTab === index ? 'primary.main' : 'text.primary' }}>
                                    {item.icon}
                                </ListItemIcon>
                                <ListItemText
                                    primary={item.text}
                                    primaryTypographyProps={{
                                        fontWeight: activeTab === index ? 'bold' : 'regular',
                                    }}
                                />
                            </ListItem>
                        ))}
                    </List>
                </Box>
            </Drawer>
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    p: 3,
                    backgroundColor: 'background.default',
                    minHeight: '100vh',
                }}
            >
                <Toolbar />
                <Paper
                    elevation={3}
                    sx={{
                        p: 3,
                        borderRadius: 2,
                        backgroundColor: 'background.paper',
                    }}
                >
                    <Stack spacing={3}>
                        <Fade in={true} timeout={1000}>
                            <Slide direction="up" in={true} mountOnEnter unmountOnExit>
                                <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
                                    {activeTab === 0 && (
                                        <>
                                            <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                                                <Paper sx={{ flexGrow: 1, overflow: 'auto', mb: 2, p: 2 }}>
                                                    <List>
                                                        {messages.map((message, index) => (
                                                            <ListItem key={index} alignItems="flex-start">
                                                                <ListItemText
                                                                    primary={message.role === 'user' ? 'Vous' : 'IA'}
                                                                    secondary={<MessageContent content={message.content} />}
                                                                />
                                                            </ListItem>
                                                        ))}
                                                        <div ref={messagesEndRef} />
                                                    </List>
                                                </Paper>
                                                <Box component="form" onSubmit={(e) => {
                                                    e.preventDefault();
                                                    const inputValue = inputRef.current ? inputRef.current.value.trim() : '';
                                                    if (inputValue) {
                                                        handleSubmit(e, inputValue);
                                                        if (inputRef.current) {
                                                            inputRef.current.value = '';
                                                        }
                                                    }
                                                }} sx={{ display: 'flex', position: 'sticky', bottom: 0, bgcolor: 'background.paper' }}>
                                                    <TextField
                                                        fullWidth
                                                        placeholder="Tapez votre message ici..."
                                                        variant="outlined"
                                                        disabled={loading}
                                                        inputRef={inputRef}
                                                    />
                                                    <Button
                                                        type="submit"
                                                        variant="contained"
                                                        disabled={loading}
                                                    >
                                                        Envoyer
                                                    </Button>
                                                </Box>
                                                <input
                                                    type="file"
                                                    accept=".pdf"
                                                    onChange={handlePDFUpload}
                                                    style={{ display: 'none' }}
                                                    id="pdf-upload"
                                                />
                                                <label htmlFor="pdf-upload">
                                                    <Button variant="contained" component="span" color="secondary" sx={{ mt: 2 }}>
                                                        Télécharger PDF
                                                    </Button>
                                                </label>
                                                <Typography variant="body2" sx={{ mt: 1, mb: 2 }}>
                                                    Vous pouvez aussi glisser-déposer un fichier PDF ici
                                                </Typography>
                                            </Box>
                                        </>
                                    )}
                                    {activeTab === 1 && (
                                        <>
                                            <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                                                <Paper sx={{ flexGrow: 1, overflow: 'auto', mb: 2, p: 2 }}>
                                                    <List>
                                                        {messages.map((message, index) => (
                                                            <ListItem key={index} alignItems="flex-start">
                                                                <ListItemText
                                                                    primary={message.role === 'user' ? 'Vous' : 'IA'}
                                                                    secondary={<MessageContent content={message.content} graphs={message.graphs} />}
                                                                />
                                                            </ListItem>
                                                        ))}
                                                        <div ref={messagesEndRef} />
                                                    </List>
                                                </Paper>
                                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2, mb: 2 }}>
                                                    {[
                                                        { key: 'document', name: 'Document' },
                                                        { key: 'sentiment', name: 'Sentiment' },
                                                        { key: 'financial_modeling', name: 'Modélisation Financière' },
                                                        { key: 'portfolio_optimization', name: 'Optimisation de Portefeuille' },
                                                        { key: 'risk_management', name: 'Gestion des Risques' },
                                                        { key: 'reporting', name: 'Rapports' },
                                                        { key: 'compliance', name: 'Conformité' },
                                                        { key: 'market_sentiment', name: 'Sentiment du Marché' },
                                                        { key: 'user_profile_analysis', name: 'Analyse du Profil Utilisateur' },
                                                        { key: 'historical_data_analysis', name: 'Analyse de Données Historiques' },
                                                        { key: 'investment_recommendation', name: 'Recommandation d\'Investissement' }
                                                    ].map((agent) => (
                                                        <Button
                                                            key={agent.key}
                                                            onClick={() => {
                                                                console.log("Agent cliqué:", agent.key);
                                                                console.log("Valeur actuelle de l'input:", agentInputRef.current ? agentInputRef.current.value : 'Pas de valeur');
                                                                handleAgentCall(agent.key);
                                                            }}
                                                            variant="contained"
                                                            color="primary"
                                                            disabled={loading}
                                                        >
                                                            {agent.name}
                                                        </Button>
                                                    ))}
                                                </Box>
                                                <Box component="form" onSubmit={(e) => {
                                                    e.preventDefault();
                                                    const inputValue = agentInputRef.current ? agentInputRef.current.value.trim() : '';
                                                    if (inputValue) {
                                                        handleSubmit(e, inputValue);
                                                        if (agentInputRef.current) {
                                                            agentInputRef.current.value = '';
                                                        }
                                                    }
                                                }} sx={{ display: 'flex', position: 'sticky', bottom: 0, bgcolor: 'background.paper' }}>
                                                    <TextField
                                                        fullWidth
                                                        placeholder="Tapez votre message ici..."
                                                        variant="outlined"
                                                        disabled={loading}
                                                        inputRef={agentInputRef}
                                                        onChange={(e) => {
                                                            if (agentInputRef.current) {
                                                                agentInputRef.current.value = e.target.value;
                                                            }
                                                        }}
                                                    />
                                                    <Button
                                                        type="submit"
                                                        variant="contained"
                                                        disabled={loading}
                                                    >
                                                        Envoyer
                                                    </Button>
                                                </Box>
                                            </Box>
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
                                                    Télécharger PDF
                                                </Button>
                                            </label>
                                            {file && <Typography variant="body1" sx={{ mt: 2 }}>{file.name}</Typography>}
                                            {loading && <CircularProgress sx={{ mt: 2 }} />}
                                        </>
                                    )}
                                    {activeTab === 3 && <Settings onClearChat={clearChatHistory} />}
                                    {activeTab === 4 && <Portfolio />}
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

                                    <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                                        <Button onClick={() => setOpenBacktest(true)} variant="outlined">Lancer Backtest</Button>
                                        <Button onClick={() => setOpenComparison(true)} variant="outlined">Comparer avec Benchmark</Button>
                                    </Box>

                                    <Dialog
                                        open={openBacktest}
                                        onClose={() => setOpenBacktest(false)}
                                        fullWidth
                                        maxWidth="lg"
                                    >
                                        <DialogTitle>Résultats du Backtest</DialogTitle>
                                        <DialogContent>
                                            <TextField
                                                label="Date de début"
                                                type="date"
                                                value={startDate}
                                                onChange={(e) => setStartDate(e.target.value)}
                                                InputLabelProps={{ shrink: true }}
                                                fullWidth
                                                margin="normal"
                                            />
                                            <TextField
                                                label="Date de fin"
                                                type="date"
                                                value={endDate}
                                                onChange={(e) => setEndDate(e.target.value)}
                                                InputLabelProps={{ shrink: true }}
                                                fullWidth
                                                margin="normal"
                                            />
                                            <Button onClick={runBacktest} variant="contained" sx={{ mt: 2 }}>Lancer Backtest</Button>
                                            {backtestResults && (
                                                <Box sx={{ mt: 2 }}>
                                                    <Typography>Rendement Total : {(backtestResults.total_return * 100).toFixed(2)}%</Typography>
                                                    <Typography>Rendement Annualisé : {(backtestResults.annualized_return * 100).toFixed(2)}%</Typography>
                                                    <Typography>Volatilité : {(backtestResults.volatility * 100).toFixed(2)}%</Typography>
                                                    <Typography>Ratio de Sharpe : {backtestResults.sharpe_ratio.toFixed(2)}</Typography>
                                                    <Box sx={{ width: '100%', height: 400, mt: 2 }}>
                                                        <ResponsiveContainer width="100%" height="100%">
                                                            <LineChart data={backtestResults.portfolio_values.map((value, index) => ({ date: index, value }))}>
                                                                <XAxis
                                                                    dataKey="date"
                                                                    tickFormatter={(tick) => {
                                                                        const date = new Date(startDate);
                                                                        date.setDate(date.getDate() + tick);
                                                                        return date.toLocaleDateString();
                                                                    }}
                                                                    interval={Math.floor(backtestResults.portfolio_values.length / 5)}
                                                                />
                                                                <YAxis
                                                                    domain={['dataMin', 'dataMax']}
                                                                    tickFormatter={(value) => `$${value.toLocaleString()}`}
                                                                />
                                                                <CartesianGrid strokeDasharray="3 3" />
                                                                <Tooltip
                                                                    formatter={(value) => [`$${value.toLocaleString()}`, "Portfolio Value"]}
                                                                    labelFormatter={(label) => {
                                                                        const date = new Date(startDate);
                                                                        date.setDate(date.getDate() + label);
                                                                        return date.toLocaleDateString();
                                                                    }}
                                                                />
                                                                <Legend />
                                                                <Line type="monotone" dataKey="value" stroke="#8884d8" dot={false} name="Portfolio Value" />
                                                            </LineChart>
                                                        </ResponsiveContainer>
                                                    </Box>
                                                </Box>
                                            )}
                                        </DialogContent>
                                        <DialogActions>
                                            <Button onClick={() => setOpenBacktest(false)}>Fermer</Button>
                                        </DialogActions>
                                    </Dialog>

                                    <Dialog
                                        open={openComparison}
                                        onClose={() => setOpenComparison(false)}
                                        fullWidth
                                        maxWidth="lg"
                                    >
                                        <DialogTitle>Comparaison de Portefeuille</DialogTitle>
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
                                                label="Date de début"
                                                type="date"
                                                value={comparisonStartDate}
                                                onChange={(e) => setComparisonStartDate(e.target.value)}
                                                InputLabelProps={{ shrink: true }}
                                                fullWidth
                                                margin="normal"
                                            />
                                            <TextField
                                                label="Date de fin"
                                                type="date"
                                                value={comparisonEndDate}
                                                onChange={(e) => setComparisonEndDate(e.target.value)}
                                                InputLabelProps={{ shrink: true }}
                                                fullWidth
                                                margin="normal"
                                            />
                                            {comparisonResults && (
                                                <Box sx={{ mt: 2 }}>
                                                    <Typography>Rendement du Portefeuille : {(comparisonResults.portfolio_return * 100).toFixed(2)}%</Typography>
                                                    <Typography>Rendement du Benchmark : {(comparisonResults.benchmark_return * 100).toFixed(2)}%</Typography>
                                                    <Typography>Volatilité du Portefeuille : {(comparisonResults.portfolio_volatility * 100).toFixed(2)}%</Typography>
                                                    <Typography>Volatilité du Benchmark : {(comparisonResults.benchmark_volatility * 100).toFixed(2)}%</Typography>
                                                    <Typography>Ratio de Sharpe du Portefeuille : {comparisonResults.portfolio_sharpe?.toFixed(2) || 'N/A'}</Typography>
                                                    <Typography>Ratio de Sharpe du Benchmark : {comparisonResults.benchmark_sharpe?.toFixed(2) || 'N/A'}</Typography>
                                                    {comparisonResults.portfolio_cumulative && comparisonResults.benchmark_cumulative && (
                                                        <Box sx={{ width: '100%', height: 400, mt: 2 }}>
                                                            <ResponsiveContainer width="100%" height="100%">
                                                                <LineChart data={comparisonResults.portfolio_cumulative.map((value, index) => ({
                                                                    date: index,
                                                                    portfolio: value,
                                                                    benchmark: comparisonResults.benchmark_cumulative[index]
                                                                }))}>
                                                                    <XAxis
                                                                        dataKey="date"
                                                                        tickFormatter={(tick) => {
                                                                            const date = new Date(comparisonStartDate);
                                                                            date.setDate(date.getDate() + tick);
                                                                            return date.toLocaleDateString();
                                                                        }}
                                                                        interval={Math.floor(comparisonResults.portfolio_cumulative.length / 5)}
                                                                    />
                                                                    <YAxis
                                                                        tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                                                                    />
                                                                    <CartesianGrid strokeDasharray="3 3" />
                                                                    <Tooltip
                                                                        formatter={(value) => [`${(value * 100).toFixed(2)}%`, ""]}
                                                                        labelFormatter={(label) => {
                                                                            const date = new Date(comparisonStartDate);
                                                                            date.setDate(date.getDate() + label);
                                                                            return date.toLocaleDateString();
                                                                        }}
                                                                    />
                                                                    <Legend />
                                                                    <Line type="monotone" dataKey="portfolio" stroke="#8884d8" dot={false} name="Portefeuille" />
                                                                    <Line type="monotone" dataKey="benchmark" stroke="#82ca9d" dot={false} name="Benchmark" />
                                                                </LineChart>
                                                            </ResponsiveContainer>
                                                        </Box>
                                                    )}
                                                </Box>
                                            )}
                                        </DialogContent>
                                        <DialogActions>
                                            <Button onClick={compareWithBenchmark} variant="contained" color="primary">Comparer</Button>
                                            <Button onClick={() => setOpenComparison(false)}>Fermer</Button>
                                        </DialogActions>
                                    </Dialog>
                                </Paper>
                            </Slide>
                        </Fade>
                    </Stack>
                </Paper>
            </Box>
            <Menu
                id="menu-appbar"
                anchorEl={anchorEl}
                anchorOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                }}
                keepMounted
                transformOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                }}
                open={Boolean(anchorEl)}
                onClose={handleClose}
            >
                <MenuItem onClick={handleClose}>Profil</MenuItem>
                <MenuItem onClick={handleClose}>Mon compte</MenuItem>
                <MenuItem onClick={handleLogout}>Déconnexion</MenuItem>
            </Menu>
        </Box>
    );
}


export default App;