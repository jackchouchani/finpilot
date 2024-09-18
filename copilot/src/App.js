import React, { useState, useEffect, useCallback } from 'react';
import { ThemeProvider, createTheme, styled, alpha } from '@mui/material/styles';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import {
    Container, TextField, Button, Paper, Typography, List, ListItem, ListItemText,
    Tab, Tabs, CircularProgress, Box, Fade, Slide, Dialog, DialogTitle,
    DialogContent, DialogActions, Select, MenuItem, AppBar, Switch,
    FormControlLabel, Toolbar, IconButton, Drawer, ListItemIcon, InputBase,
    Badge, Menu, useMediaQuery, CssBaseline
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

function a11yProps(index) {
    return {
        id: `scrollable-auto-tab-${index}`,
        'aria-controls': `scrollable-auto-tabpanel-${index}`,
    };
}

const drawerWidth = 240;

const Root = styled('div')(({ theme }) => ({
    display: 'flex',
}));

const AppBarStyled = styled(AppBar, {
    shouldForwardProp: (prop) => prop !== 'open',
})(({ theme, open }) => ({
    zIndex: theme.zIndex.drawer + 1,
    transition: theme.transitions.create(['width', 'margin'], {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.leavingScreen,
    }),
    ...(open && {
        marginLeft: drawerWidth,
        width: `calc(100% - ${drawerWidth}px)`,
        transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    }),
}));

const DrawerStyled = styled(Drawer, { shouldForwardProp: (prop) => prop !== 'open' })(
    ({ theme, open }) => ({
        '& .MuiDrawer-paper': {
            position: 'relative',
            whiteSpace: 'nowrap',
            width: drawerWidth,
            transition: theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
            }),
            boxSizing: 'border-box',
            ...(!open && {
                overflowX: 'hidden',
                transition: theme.transitions.create('width', {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.leavingScreen,
                }),
                width: theme.spacing(7),
                [theme.breakpoints.up('sm')]: {
                    width: theme.spacing(9),
                },
            }),
        },
    }),
);

const SearchIconWrapper = styled('div')(({ theme }) => ({
    padding: theme.spacing(0, 2),
    height: '100%',
    position: 'absolute',
    pointerEvents: 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
}));

const Search = styled('div')(({ theme }) => ({
    position: 'relative',
    borderRadius: theme.shape.borderRadius,
    backgroundColor: alpha(theme.palette.common.white, 0.15),
    '&:hover': {
        backgroundColor: alpha(theme.palette.common.white, 0.25),
    },
    marginRight: theme.spacing(2),
    marginLeft: 0,
    width: '100%',
    [theme.breakpoints.up('sm')]: {
        marginLeft: theme.spacing(3),
        width: 'auto',
    },
}));

const StyledInputBase = styled(InputBase)(({ theme }) => ({
    color: 'inherit',
    '& .MuiInputBase-input': {
        padding: theme.spacing(1, 1, 1, 0),
        paddingLeft: `calc(1em + ${theme.spacing(4)})`,
        transition: theme.transitions.create('width'),
        width: '100%',
        [theme.breakpoints.up('md')]: {
            width: '20ch',
        },
    },
}));

const MainContent = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })(
    ({ theme, open }) => ({
        flexGrow: 1,
        padding: theme.spacing(3),
        transition: theme.transitions.create('margin', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
        }),
        marginLeft: `-${drawerWidth}px`,
        ...(open && {
            transition: theme.transitions.create('margin', {
                easing: theme.transitions.easing.easeOut,
                duration: theme.transitions.duration.enteringScreen,
            }),
            marginLeft: 0,
        }),
    }),
);


function App() {
    const [darkMode, setDarkMode] = useState(false);
    const [open, setOpen] = useState(true);
    const [anchorEl, setAnchorEl] = useState(null);
    const [drawerOpen, setDrawerOpen] = useState(false);
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

    const handleDrawerOpen = () => {
        setOpen(true);
    };

    const handleDrawerClose = () => {
        setOpen(false);
    };

    const handleMenu = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const toggleDarkMode = () => {
        setDarkMode(!darkMode);
    };

    const theme = createTheme({
        palette: {
            mode: darkMode ? 'dark' : 'light',
            primary: {
                main: '#1976d2',
            },
            secondary: {
                main: '#dc004e',
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

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        setLoading(true);
        const newMessage = { role: 'user', content: input };
        setMessages(prevMessages => [newMessage, ...prevMessages]);
        setInput('');

        try {
            const response = await axios.post(process.env.REACT_APP_API_URL + '/chat', {
                message: input,
                conversation_id: conversationId
            });
            setConversationId(response.data.conversation_id);
            setMessages(prevMessages => [{ role: 'assistant', content: response.data.reply }, ...prevMessages]);
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
        const userMessage = { role: 'user', content: input };
        setMessages(prevMessages => [userMessage, ...prevMessages]);
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
            setMessages(prevMessages => [newMessage, ...prevMessages]);

            // Sauvegardez le message de l'agent dans la base de données
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
            setMessages(prevMessages => [errorMessage, ...prevMessages]);

            // Sauvegarder le message d'erreur dans l'historique du chat
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

    const handleLogin = async () => {
        try {
            const response = await axios.post('/login', { username, password });
            console.log(response.data.msg);
            // Rediriger ou mettre à jour l'état de l'application
        } catch (error) {
            console.error("Erreur lors de la connexion :", error);
            alert('Échec de la connexion. Vérifiez vos identifiants.');
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
            <ThemeProvider theme={theme}>
                <CssBaseline />
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
                input={input}
                setInput={setInput}
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
    input,
    setInput,
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
                    <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                        FinPilot
                    </Typography>
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
                        label="Dark Mode"
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
                    [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
                }}
            >
                <Toolbar />
                <Box sx={{ overflow: 'auto' }}>
                    <List>
                        {['Copilot', 'Agents', 'PDF Analysis', 'Settings', 'Portfolio', 'Market Sentiment', 'Investment Recommendation', 'Historical Data Analysis', 'User Profile Analysis'].map((text, index) => (
                            <ListItem button key={text} onClick={() => setActiveTab(index)}>
                                <ListItemIcon>
                                    {index === 0 && <ChatIcon />}
                                    {index === 1 && <GroupWorkIcon />}
                                    {index === 2 && <PictureAsPdfIcon />}
                                    {index === 3 && <SettingsIcon />}
                                    {index === 4 && <AccountBalanceIcon />}
                                    {index === 5 && <TrendingUpIcon />}
                                    {index === 6 && <MonetizationOnIcon />}
                                    {index === 7 && <HistoryIcon />}
                                    {index === 8 && <PersonIcon />}
                                </ListItemIcon>
                                <ListItemText primary={text} />
                            </ListItem>
                        ))}
                    </List>
                </Box>
            </Drawer>
            <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
                <Toolbar />
                <Fade in={true} timeout={1000}>
                    <Slide direction="up" in={true} mountOnEnter unmountOnExit>
                        <Container maxWidth="lg">
                            <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
                                {activeTab === 0 && (
                                    <>
                                        <List>
                                            {messages.map((message, index) => (
                                                <ListItem key={index} alignItems="flex-start">
                                                    <ListItemText
                                                        primary={message.role === 'user' ? 'You' : 'AI'}
                                                        secondary={<MessageContent content={message.content} graphs={message.graphs} />}
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
                                            <Button variant="contained" component="span" color="secondary" sx={{ mt: 2 }}>
                                                Upload PDF
                                            </Button>
                                        </label>
                                        <Typography variant="body2" sx={{ mt: 1 }}>
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
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                                            {['document', 'sentiment', 'financial_modeling', 'portfolio_optimization', 'risk_management', 'reporting', 'compliance', 'market_sentiment', 'user_profile_analysis', 'historical_data_analysis', 'investment_recommendation'].map((agent) => (
                                                <Button
                                                    key={agent}
                                                    onClick={() => handleAgentCall(agent)}
                                                    variant="contained"
                                                    color="primary"
                                                    disabled={loading}
                                                >
                                                    {agent.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                                                </Button>
                                            ))}
                                        </Box>
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
                                        {file && <Typography variant="body1" sx={{ mt: 2 }}>{file.name}</Typography>}
                                        {loading && <CircularProgress sx={{ mt: 2 }} />}
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
                                    <Button onClick={() => setOpenBacktest(true)} variant="outlined">Run Backtest</Button>
                                    <Button onClick={() => setOpenComparison(true)} variant="outlined">Compare with Benchmark</Button>
                                </Box>

                                <Dialog
                                    open={openBacktest}
                                    onClose={() => setOpenBacktest(false)}
                                    fullWidth
                                    maxWidth="lg"
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
                                        <Button onClick={runBacktest} variant="contained" sx={{ mt: 2 }}>Run Backtest</Button>
                                        {backtestResults && (
                                            <Box sx={{ mt: 2 }}>
                                                <Typography>Total Return: {(backtestResults.total_return * 100).toFixed(2)}%</Typography>
                                                <Typography>Annualized Return: {(backtestResults.annualized_return * 100).toFixed(2)}%</Typography>
                                                <Typography>Volatility: {(backtestResults.volatility * 100).toFixed(2)}%</Typography>
                                                <Typography>Sharpe Ratio: {backtestResults.sharpe_ratio.toFixed(2)}</Typography>
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
                                        <Button onClick={() => setOpenBacktest(false)}>Close</Button>
                                    </DialogActions>
                                </Dialog>

                                <Dialog
                                    open={openComparison}
                                    onClose={() => setOpenComparison(false)}
                                    fullWidth
                                    maxWidth="lg"
                                >
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
                                        {comparisonResults && (
                                            <Box sx={{ mt: 2 }}>
                                                <Typography>Portfolio Return: {(comparisonResults.portfolio_return * 100).toFixed(2)}%</Typography>
                                                <Typography>Benchmark Return: {(comparisonResults.benchmark_return * 100).toFixed(2)}%</Typography>
                                                <Typography>Portfolio Volatility: {(comparisonResults.portfolio_volatility * 100).toFixed(2)}%</Typography>
                                                <Typography>Benchmark Volatility: {(comparisonResults.benchmark_volatility * 100).toFixed(2)}%</Typography>
                                                <Typography>Portfolio Sharpe Ratio: {comparisonResults.portfolio_sharpe?.toFixed(2) || 'N/A'}</Typography>
                                                <Typography>Benchmark Sharpe Ratio: {comparisonResults.benchmark_sharpe?.toFixed(2) || 'N/A'}</Typography>
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
                                                                <Line type="monotone" dataKey="portfolio" stroke="#8884d8" dot={false} name="Portfolio" />
                                                                <Line type="monotone" dataKey="benchmark" stroke="#82ca9d" dot={false} name="Benchmark" />
                                                            </LineChart>
                                                        </ResponsiveContainer>
                                                    </Box>
                                                )}
                                            </Box>
                                        )}
                                    </DialogContent>
                                    <DialogActions>
                                        <Button onClick={compareWithBenchmark} variant="contained" color="primary">Compare</Button>
                                        <Button onClick={() => setOpenComparison(false)}>Close</Button>
                                    </DialogActions>
                                </Dialog>
                            </Paper>
                        </Container>
                    </Slide>
                </Fade>
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
                <MenuItem onClick={handleClose}>Profile</MenuItem>
                <MenuItem onClick={handleClose}>My account</MenuItem>
                <MenuItem onClick={handleLogout}>Logout</MenuItem>
            </Menu>
        </Box>
    );
}

export default App;