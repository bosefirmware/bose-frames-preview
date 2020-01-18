/* eslint-disable no-shadow,no-unused-expressions,no-undef,guard-for-in,no-empty-function */
/**
 * Simple JavaScript Inheritance
 * By John Resig http://ejohn.org/
 * MIT Licensed.
 */
// Inspired by balse2 and Prototype
(function() {
    var initializing = false,
        fnTest = (/xyz/).test(function() {
            xyz;
        }) ? /\b_super\b/ : /.*/;

    // The base Class implementation (does nothing)
    this.Class = function() {};

    // Create a new Class that inherits from this class
    Class.extend = function(prop) {
        var _super = this.prototype;

        // Instantiate a base class (but only create the instance,
        // Don't run the init constructor)
        initializing = true;
        var prototype = new this();

        initializing = false;

        // Copy the properties over onto the new prototype
        for (var name in prop) {
            // Check if we're overwriting an existing function
            prototype[name] = typeof prop[name] === 'function' &&
            typeof _super[name] === 'function' && fnTest.test(prop[name]) ?
                (function(name, fn) {
                    return function() {
                        var tmp = this._super;

                        // Add a new ._super() method that is the same method
                        // But on the super-class
                        this._super = _super[name];

                        // The method only need to be bound temporarily, so we
                        // Remove it when we're done executing
                        var ret = fn.apply(this, arguments);

                        this._super = tmp;

                        return ret;
                    };
                }(name, prop[name])) : prop[name];
        }

        // The dummy class constructor
        function Class() {
            // All construction is actually done in the init metho
            if (!initializing && this.init) {
                this.init.apply(this, arguments);
            }
        }

        // Populate our constructed prototype object
        Class.prototype = prototype;

        // Enforce the constructor to be what we expect
        Class.prototype.constructor = Class;

        // And make this class extendable
        Class.extend = arguments.callee;

        return Class;
    };
}());
/* eslint-disable no-undef,jsdoc/check-tag-names,jsdoc/require-returns-description,valid-jsdoc,jsdoc/check-types,jsdoc/require-description-complete-sentence,jsdoc/require-hyphen-before-param-description */
var Bose = Bose || {};

Bose.BaseClass = Class.extend(function() {
    'use strict';

    return {
        /**
         * @property {object} $el Reference to the jQuery element
         * @memberOf Bose.BaseClass
         */
        $el: null,
        /**
         * @property {object} cfg Original configuration
         * @memberOf Bose.BaseClass
         */
        cfg: {},
        /**
         * @property {object} el Reference to the DOM element
         * @memberOf Bose.BaseClass
         */
        el: null,
        /**
         * @property {boolean} isInitialized Current initialization status
         * @memberOf Bose.BaseClass
         */
        isInitialized: false,
        /**
         * @property {boolean} isSetup Current setup status
         * @memberOf Bose.BaseClass
         */
        isSetup: false,
        /**
         * @property {string} mod Module shortname used to initialize the instance
         * @memberOf Bose.BaseClass
         */
        mod: '',
        /**
         * @function init
         * @description Initializes the current class instance
         * @memberOf Bose.UI.BaseClass
         * @param {object} cfg Configuration object
         * @returns {void}
         */
        init: function init(cfg) {
            var _this = this;

            _this.cfg = cfg;
            $jq.extend(_this, _this.cfg);

            if (typeof _this.el === 'object' && _this.el !== null) {
                _this.el.instance = _this;
                _this.$el = $jq(this.el);
                $jq.extend(_this, _this.el.dataset);
                _this.el.classList.add('initialized');
                _this.isInitialized = true;

                _this._setup();
            }
        },
        /**
         * @function _setup
         * @description Performs all basic setup tasks for a module
         * @memberOf Bose.UI.BaseClass
         * @abstract
         * @returns {void}
         */
        _setup: function _setup() {
            var _this = this;

            _this.el.classList.add('setup');
            _this.isSetup = true;
        }
    };
}());
